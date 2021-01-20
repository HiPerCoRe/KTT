#This code is for predicting the value for profiling counter values

"""Generating train file for predicting the profiling counter values based on tuning parameters

Usage:
  generate-knowledge-base.py -i <KTT_output> -t <tuningpar_interval> -c <counters_interval>

Options:
  -h Show this screen.
  -i Output CSV file from KTT (must include profiling counters)-This file will use as an input for the model.
  -t Interval <i:j,[k,l,m]>  or i:j of tuning parameters indices (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).
  -c Interval <i:j,[k,l,m]>  or i:j of profiler counter indices (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).

"""

import os
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import datetime
from docopt import docopt



np.set_printoptions(suppress=True)
TEST_SIZE = 50
SEED = 7


if __name__ == '__main__':
    # parse command line
    arguments = docopt(__doc__)
    tuningOutput = open(arguments['-i'])
    #tuningInt = list(map(int, arguments['-t'].split(',')))
    tuningRange = arguments['-t']
    #countersInt = list(map(int, arguments['-c'].split(',')))
    countersRange = arguments['-c']
    #if (len(tuningInt) != 2) or (len(countersInt) != 2):
    #    print("Intervals must be in format from,to!")
    #    exit()

    bench = os.path.splitext(arguments['-i'])[0]
    data = pd.read_csv(tuningOutput)
    array = data.values

    try:
        rangeT = eval("np.r_[" + tuningRange + "]")
        rangeC = eval("np.r_[" + countersRange + "]")
    except:
        print("Intervals must be in format <i:j,[k,l,m]> or i:j (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).")
        exit()
    #array[1:2,np.r_[1:2,7:10,[13,14]]]
    #X = array[:,tuningInt[0]:tuningInt[1]+1]
    X = array[:,rangeT]
    #Y = array[:,countersInt[0]:countersInt[1]]  # Using profiling counter variables as dependent variables
    Y = array[:,rangeC]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(TEST_SIZE/100), random_state=SEED)
    columns_names = list(data.columns)
    columns_number = len(data.columns)
    # Estimate the score after iterative imputation of the missing values
    # with different estimators
    estimators = []
    estimators.append(('DT', DecisionTreeRegressor(max_features='sqrt', random_state=0)))
    estimators.append(('Proposed', ExtraTreesRegressor(n_estimators=10, random_state=0)))
    estimators.append(('Knn10', KNeighborsRegressor(n_neighbors=10)))

    NumberOfModels = len(estimators)
    ExprimentStart = datetime.datetime.now()
    start = 0
    end = 0
    maxScore = 0
    Bestmodel = ''
    for name, imputer in estimators:
        start = datetime.datetime.now()
        imputer.fit(X_train, Y_train)
        #saving model to a file for later usages
        filename = str(bench) + "_" + str(name) + ".sav"
        pickle.dump(imputer, open(filename, 'wb'))
        #Computing the score of model
        scoreTrain = imputer.score(X_train, Y_train)
        #compute time elapsed
        end = datetime.datetime.now()
        durationTrain = end - start
        #Predicting with Test or X_test dataset and then writing the predicted results to .csv
        start = datetime.datetime.now()
        predicted = imputer.predict(X_test)
        scoreTest = imputer.score(X_test, Y_test)
        if (scoreTest >= maxScore):
            maxScore = scoreTest
            Bestmodel = name
        end = datetime.datetime.now()
        durationTest = end - start
        #Print some reports
        print("Training Result for ", name, " and ", bench, " is :")
        print("             Train Score is                : %", scoreTrain * 100)
        print("             Test Score is                 : %", scoreTest * 100)
        print("             Mean squared error is         : ", mean_squared_error(Y_test, predicted))
        print("=======================================================================")

        # Save list of profiling counters
        filename = filename + ".pc"
        pcFile = open(filename, "w")
        pcFile.write('Profiling counter\n')
        for col in rangeC:
            pcFile.write(str(data.columns[col]) + "\n")
        pcFile.close()


    ExprimentEnd = datetime.datetime.now()
    TotalTime = ExprimentEnd - ExprimentStart
    print(Bestmodel, ' : For ', str(bench), ' and test size ', TEST_SIZE, ' === Maximum score between these models are:', maxScore)
    print("Total time that elapsed in this expriment is: ", TotalTime.total_seconds() * 1000)
    print("=======================================================================")
