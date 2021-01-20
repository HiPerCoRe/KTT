
  ### HELPFUL FUNCTIONS

  getRealValue <- function(codedValue, centerValue, halfr) {
    codedValue*halfr + centerValue
  }
  
  getRealValueRow <- function(codedValueRow, centerValues, halfRange) {
    mapply(getRealValue, codedValueRow, centerValues, halfRange)
  }
  
  getRealValueVector <- function(codedVector, name, centerValues, halfRange) {
    sapply(codedVector, getRealValue, centerValue = centerValues[name], halfr = halfRange[name])
  }
  
  getCodedValue <- function(realValue, centerValue, halfr) {
    if (halfr != 0)
      (realValue-centerValue)/halfr
    else
      -1
  }
  
  getCodedValueRow <- function(realV, centerValues, halfRange) {
    mapply(getCodedValue, realV, centerValues, halfRange)
  }
  
  getCodedValueVector <- function(realVector, name, centerValues, halfRange) {
    sapply(realVector, getCodedValue, centerValue = centerValues[name], halfr = halfRange[name])
  }
  
  updateCodedValues <- function(listOfObjects, nnames) {
    defaultW <- getOption("warn")
    options(warn = -1)
    m <- as.matrix(as.data.frame(listOfObjects))
    colnames(m) <- nnames
    options(warn = defaultW)
    m
  }
  
  updateRealValues <- function(codedValues, nnames, centerValues, halfRange) {
    m <- t(apply(codedValues, MARGIN=1, getRealValueRow, centerValues, halfRange))
    colnames(m) <- nnames
    m
  }
  
  # if a tuning parameter has only one possible value, this value is its center value and halfRange is 0
  # if a tuning parameter has two possible values, the halfRange is mean and center values is max - halfRange
  determineCenterValuesAndHalfRangeForBinaryValues <- function(data, factorNames, possibleValues) {
    centerValues <- rep(NA, length(factorNames))
    names(centerValues) <- names(factorNames)
    halfRange <- rep(NA, length(factorNames))
    names(halfRange) <- names(factorNames)
    for (f in names(factorNames)) {
      if (length(possibleValues[[f]]) ==  1) {
        halfRange[f] <- 0
        centerValues[f] <- possibleValues[[f]][1]
        
      }
      if (length(possibleValues[[f]]) ==  2) {
        halfRange[f] <- (max(possibleValues[[f]]) - min(possibleValues[[f]]))/2
        centerValues[f] <- max(possibleValues[[f]] - halfRange[f])
        
      }
    }
    list(centerValues, halfRange)
  }
  
  determineCenterValuesAndHalfRange <- function(data, centerValues, halfRange, factorNames, possibleValues) {
    for (f in names(factorNames)) {
      # disregard those that are already done, i.e. single-value and binary parameters
      if (!is.na(centerValues[[f]]))
        next
      # otherwise center value is median
      halfRange[f] <- (median(possibleValues[[f]]) - min(possibleValues[[f]]))
      centerValues[f] <- median(possibleValues[[f]])
    }
    list(centerValues, halfRange)
  }
  
  getBasicCodedValues <- function(possibleValues, centerValues, halfRange, name) {
    getCodedValueVector(possibleValues[[name]], name, centerValues, halfRange)
  }
  
  selectValues <- function(v, takeMore) {
    if (takeMore) {
      if (length(v) <= 3)
        v
      else if (length(v) == 4)
        v[c(1,2,4)]
      else if (length(v) == 5)
        v[c(1,3,5)]
      else if (length(v) == 6)
        v[c(2,4,6)]
      else if (length(v) == 7)
        v[c(2,4,6)]
      else if (length(v) == 8)
        v[c(3,5,7)]
      else
        c(v[length(v)/4+1],v[length(v)/2+1], v[length(v)*3/4+1])
    }
    else {
      if (length(v) == 3)
        v[c(1,3)]
      else if (length(v) == 4)
        v[c(2,4)]
      else if (length(v) == 5)
        v[c(1,5)]
      else if (length(v) == 6)
        v[c(2,6)]
      else if (length(v) == 7)
        v[c(2,6)]
      else if (length(v) == 8)
        v[c(3,7)]
      else
        c(v[length(v)/4+1], v[length(v)*3/4+1])
    }
  }

  selectDataPointsFromTuningSpace <- function(tuningSpace, middlePoint, binaryFactors, nonBinaryFactors, selectedValues) {

    condition <- rep(TRUE, length.out = nrow(tuningSpace))
    for (a in names(binaryFactors)) {
      condition <- condition & (tuningSpace[,binaryFactors[a]] == middlePoint[a])
    }
    for (a in names(nonBinaryFactors)) {
      condition <- condition & (tuningSpace[,nonBinaryFactors[a]] %in% selectedValues[[a]])
    }
    tuningSpace[condition,]
  }

  
  findPossibleValues <- function(data, factorNames, nnames) {
    l <- list()
    for (i in 1:length(factorNames)) {
      l[[nnames[i]]] <- sort(unique(data[,factorNames[i]]))
    }
    l
  }


  getOneOutput <- function(data, outputName, realRow, factorNames) {
    
    condition <- rep(TRUE, length.out = nrow(data))
    for (a in names(factorNames)) {
      condition <- condition & (data[,factorNames[a]] == as.numeric(realRow[a]))
    }
    
    data[
      condition
      ,outputName]
  }
  
  
  getAllOutput <- function(data, outputName, realValues, factorNames) {
    y <- c()
    for (i in 1:nrow(realValues)) {
      realRow <- realValues[i,]
     # names(realRow) <- names(factorNames)
      v <- getOneOutput(data, outputName, realRow, factorNames)
      if (length(v) == 0)
        y <- c(y,NA)
      else if (length(v) > 1)
        y <- c(y, sample(v,1))
      else
        y <- c(y,v)
    }
    y
  }
  
  printModel <- function(model, outputVariable) {
    co <- model$coefficients[!is.na(model$coefficients)]
    #replace I(something^2) with (something^2) 
    for (i in 1:length(names(co))) {
      name <- names(co)[i]
      if (startsWith(name, "I("))
        names(co)[i] <- gsub('\\^', '**', substr(name, 2, nchar(name)))
    }
    
    #replace : with * in interaction terms names  
    names(co) <- sapply(names(co), FUN=gsub, pattern=':', replacement='*')
    
    paste0(round(co[1],2), "",
           paste(sprintf(" %+.2f*%s ",
                         co[-1],names(co[-1])),
                 collapse=""))
  }
  
  writeModelToFile <- function(outputFile, factorNames, codedFactorNames, codedValues, centerValues, halfRange, outputVariables, listModels) {
    all <- matrix(nrow=length(factorNames) + 1 + length(outputVariables), ncol=2)
    row <- 1
    for (f in factorNames) {
      g <- codedFactorNames[f]
      if (halfRange[g] == 0)
        all[row,] <- c(f, paste0(g, "=-1"))
      else
        all[row,] <- c(f, paste0(g,"=(",f,"-",centerValues[g],")/",halfRange[g]))
      row <- row+1
    }
    condition <- paste0(codedFactorNames[binaryFactors[1]],"==",codedValues[1,codedFactorNames[binaryFactors[1]]])
    for (f in binaryFactors[-1]) {
      condition <- paste0(condition, " and ", codedFactorNames[f], "==", codedValues[1,codedFactorNames[f]])
    }
    all[row,] <- c("Condition", condition)
    row <- row+1
    
    for (i in 1:length(outputVariables)) {
      all[row,] <- c(outputVariables[i], printModel(listModels[[outputVariables[i]]]))
      row <- row+1
    }
    
    df <- data.frame(all, stringsAsFactors=FALSE)
    write.csv(df, file=outputFile, row.names = FALSE)
    
  }
  
  parseRange <- function(range) {
    columns <- c()
    cs <- unlist(strsplit(range, ","))
    for (i in cs) {
      s <- as.numeric(unlist(strsplit(i, ":")))
      if (length(s) == 1)
        columns <- c(columns, s)
      else
        columns <- c(columns, seq(from = s[1]+1, to = s[2]))
    }
    columns

  }


  ### CODE STARTS HERE
  ## First, some background knowledge necessary to understand the script.

  # Non-linear models are created with R function lm while including non-linear (quadratic) terms in formula.
  # lm takes formula in format y ~ terms where terms include the factors and arithmetic operations with them.
  # When non-linear terms are included, such as quadratic terms, or term where several factors are multiplied
  #  (i.e. most commonly two-way or three-way interactions), the resulting model is non-linear.
  # Moreover, lm takes datapoints (training data) for y (in our case, the given profiling counter)

  # Models in general do not work well with absolute values of the factors. It is highly recommended to
  #  "code" them, i.e. interpolate values to the range of <-1,1>.

  # The creation of non-linear models follows a rather simple procedure.
  # - parse the command line options
  #   - input file with tuning space and their evaluated performance and profiling counters
  #   - output file name, this is then concatenated with -model_[number].csv
  #   - numbers of columns with tuning parameters in format allowing , and :, e.g. 2,5:12 meaning columns 2 and 5 through 12
  #   - numbers of columns with profiling counters in the same format
  # - code the values of factors (tuning parameters), this encompasses
  #   - finding possible values of the tuning parameters
  #   - finding center values and halves of the range for all of them to be able to easily switch between coded and decoded values
  #   - assigning all values of all single-value and binary parameters to coded variables named by letters from alphabet
  #   - assigning two or three values of other parameters to coded variables, this needs to deal with
  #       - in some cases, the number of datapoints increases exponentially, we need to limit this
  #       - in other cases, constraints can cause no or little datapoints, we need to prevent that we won't have enough datapoints because of this
  # - create the model for each combination of values of binary parameters
  #   - and for each profiling counter

  # The output of the script are multiple files named [output_name]-model_[number].csv.
  # The number of models corresponds to number of combinations of values of binary parameters
  #    (minus ones that are not allowed by constraints).
  # Each file includes
  #   - for each profiling counter an expression that can be used to predict its value
  #         based on coded values of tuning parameters
  #   - expressions to code each tuning parameter
  #   - condition that is true for that given combination of values of binary parameters this model was trained for


  # parse the command line options
  args = commandArgs(trailingOnly=TRUE)


  #args <- c("/home/janka/research/autotuning/autotuning/profilbased-searcher/data-reducedcounters/1070-conv_output.csv", "1070-conv", "4,13", "14,56")
  data <- read.csv(args[1])

  outputFile <- args[2]
  factorColumnsString <- args[3]
  factorColumns <- parseRange(factorColumnsString)
  profCountColumnsString <- args[4]
  profCountColumns <- parseRange(profCountColumnsString)
  
  trainingPortion <- 0.4
  trainingSize <- floor(nrow(data)*trainingPortion)
  factorNames <- colnames(data)[factorColumns]
  abcd <- c("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "Z")
  if (length(factorNames) > length(abcd)) {
    quit(status = 1)
  }
    
  nnames <- abcd[1:length(factorNames)]
  names(factorNames) <- nnames
  
  codedFactorNames <- nnames
  names(codedFactorNames) <- factorNames
  
  # code the values of factors (tuning parameters)
  # finding possible values of the tuning parameters
  possibleValues <- findPossibleValues(data, factorNames, nnames)
  
  # finding center values and halves of the range for all of them to be able to easily switch between coded and decoded values
  ll <- determineCenterValuesAndHalfRangeForBinaryValues(data, factorNames, possibleValues)
  centerValues <- ll[[1]]
  halfRange <- ll[[2]]
  binaryFactors <- factorNames[!is.na(centerValues)]
  nonBinaryFactors <- factorNames[is.na(centerValues)]
  
  ll <- determineCenterValuesAndHalfRange(data, centerValues, halfRange, factorNames, possibleValues)
  centerValues <- ll[[1]]
  halfRange <- ll[[2]]
  
  basicCodedValues <- list() #what are coded values for all possible values of tuning parameters?
  selectedValues <- list()

  for (x in nnames) {
    basicCodedValues[[x]] <- getCodedValueVector(possibleValues[[x]], x, centerValues, halfRange)
  }
  

  # assigning all values of all single-value and binary parameters to coded variables named by letters from alphabet
  for (x in names(binaryFactors))
    assign(x, basicCodedValues[[x]])
  

  designBinaryFactors <- expand.grid(mget(names(binaryFactors)))

  middlePoint <- mapply(`[[`, possibleValues, lengths(basicCodedValues)/2+1)
  
  outputVariables <- colnames(data)[profCountColumns]
  
  listModels <- list()
  training <- 0
  trainingThis <- trainingSize/nrow(designBinaryFactors)
  # we create a model for each combination of binary parameters' values as we assume that
  # these and their combinations influence the code performance significantly
  for (j in 1:nrow(designBinaryFactors))
  {
    # we could not assign all values of multiple-value parameters to coded variables and just use as is
    # because the number of configurations (combinations of values of parameters) explodes exponentially
    # but, in order to have a model that somewhat well describes the tuning space
    # we need a reasonable number of reasonably well-spaced datapoints to train the model
    # moreover, we need to deal with constraints: not all combinations of parameters' values are valid
    # even though the rules for constraints are known beforehand,
    # putting them here would require changes in code done by user and specifically for each problem
    # so, instead we iteratively try to add two points of non-binary factors
    # check if the datapoints with their values and values of other parameters are present in input file
    # if so, we add them, if not, we try adding three points to increase the chance to avoid constraints

    for (x in names(binaryFactors))
      middlePoint[x] <- getRealValue(designBinaryFactors[j,x], centerValues[x], halfRange[x])
    
    level <- 0

    repeat {
      if (level == 0)
        takeMore <- NULL
      else
        takeMore <- names(possibleValues)[order(lengths(possibleValues), decreasing = TRUE)[1:level]]
      for (x in nnames) {
        if (x %in% takeMore)
          selectedValues[[x]] <- selectValues(possibleValues[[x]], TRUE)
        else
          selectedValues[[x]] <- selectValues(possibleValues[[x]], FALSE)
      }

      designTry <- selectDataPointsFromTuningSpace(data[,factorColumns], middlePoint, binaryFactors, nonBinaryFactors, selectedValues)

      if (nrow(designTry) > trainingThis | level == length(nnames))
        break
      else {
        design <- designTry
        level <- level + 1
      }
    }


    training <- training+nrow(design)

    realValues <- design
    names(realValues) <- nnames
    
    for (a in nnames)
      assign(a, getCodedValueVector(design[factorNames[a]], a, centerValues, halfRange))
    
    codedValues <- updateCodedValues(mget(nnames), nnames = nnames)
    # at this point, we have coded values of parameters' values for training in codedValues and in variables
    # named by letters of alphabet
    
    listModelsPC <- list()
    # for each profiling counter
    for (i in 1:length(outputVariables)) {
      # gather all measurements of this profiling counter for configurations that correspond to
      # decoded codedValues
      y <- getAllOutput(data, outputVariables[i], realValues, factorNames)
      if (all(is.na(y)))
        break
      # create the formula
      formula <- "y ~ "
      #add an expression of high-order interactions, e.g. A*B*C, lm itself decomposes this to
      # main effects (i.e. A+B+C) and interaction terms (+A*B+A*C+B*C+A*B*C)
      formula <- paste0(formula, names(nonBinaryFactors)[1])
      formula <- paste0(formula, paste(sprintf("*%s", names(nonBinaryFactors)[-1]),collapse=""))
      #add quadratic terms to include non-linear effects
      formula <- paste0(formula, paste(sprintf(" + I(%s^2)", names(nonBinaryFactors)),collapse=""))
      # create the model
      ml <- lm(as.formula(formula))
      listModelsPC[[outputVariables[i]]] <- ml
      #printModel(ml, outputVariables[i])
      #print(paste("Calculation", j, "variable", i, "done"))
    }
    if (all(is.na(y)))
      next
    listModels[[j]] <- list(designBinaryFactors[j, names(binaryFactors)],listModelsPC)
    # write the model to a csv file
    writeModelToFile(paste0(outputFile,"-model_",j,".csv"), factorNames, codedFactorNames, codedValues, centerValues, halfRange, outputVariables, listModelsPC)
  }
  print(paste("Used", training, "datapoints out of", nrow(data), "to train"))
  
#save(factorNames, centerValues, halfRange, listModels, file="models.RData")