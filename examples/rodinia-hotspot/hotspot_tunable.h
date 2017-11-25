#pragma once

#include "tuner_api.h"
#include "hotspot.h"
#include "hotspot_reference.h"

class tunableHotspot : public ktt::TuningManipulator {
  public:

    //Constructor creates internal structures and setups the tuning environment
    // it takes arguments passed from command line arguments and initializes all data variables
    
    tunableHotspot(ktt::Tuner *tuner, std::string kernelFile, int grid_rows, int grid_cols, int pyramid_height, int total_iterations, char* tfile, char* pfile, char *ofile) : TuningManipulator() {

      this->tuner = tuner;

      // Initialize data variables
      // we need some of them calculated to set the ndRange for example
      this->grid_rows = grid_rows;
      this->grid_cols = grid_cols;
      this->pyramid_height = pyramid_height;
      this->total_iterations = total_iterations;
      this->ofile = ofile;
      size = grid_rows*grid_cols;
      // --------------- pyramid parameters --------------- 
      borderCols = (pyramid_height)*EXPAND_RATE/2;
      borderRows = (pyramid_height)*EXPAND_RATE/2;
      smallBlockCol = BLOCK_SIZE_REF-(pyramid_height)*EXPAND_RATE;
      smallBlockRow = BLOCK_SIZE_REF-(pyramid_height)*EXPAND_RATE;
      blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
      blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
      grid_height = chip_height / grid_rows;
      grid_width = chip_width / grid_cols;

      Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
      Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
      Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
      Rz = t_chip / (K_SI * grid_height * grid_width);

      max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
      step = PRECISION / max_slope / 100.0;


      tempSrc = std::vector<float>(size);
      tempDst = std::vector<float>(size);
      power = std::vector<float>(size);

      // Read input data from disk
      readinput(tempSrc, grid_rows, grid_cols, tfile);
      readinput(power, grid_rows, grid_cols, pfile);

      // Declare kernel parameters
      // Total NDRange size matches number of grid points
      const ktt::DimensionVector ndRangeDimensions(1, 1, 1);
      const ktt::DimensionVector workGroupDimensions(1, 1, 1);
      // Add two kernels to tuner, one of the kernels acts as reference kernel
      // 
      kernelId = tuner->addKernelFromFile(kernelFile, std::string("hotspot"), ndRangeDimensions, workGroupDimensions);
      // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
      tuner->addParameter(kernelId, "BLOCK_SIZE", {4, 8, 16, 32});
      // Add conditions
      // so far none

      // Add all arguments utilized by kernels
      iterationId = tuner->addArgument(iteration);
      size_t powerId = tuner->addArgument(power, ktt::ArgumentMemoryType::ReadOnly);
      tempSrcId = tuner->addArgument(tempSrc, ktt::ArgumentMemoryType::ReadWrite);
      tempDstId = tuner->addArgument(tempDst, ktt::ArgumentMemoryType::ReadWrite);
      size_t grid_colsId = tuner->addArgument(grid_cols);
      size_t grid_rowsId = tuner->addArgument(grid_rows);

      size_t borderColsId = tuner->addArgument(borderCols);
      size_t borderRowsId = tuner->addArgument(borderRows);
      size_t CapId = tuner->addArgument(Cap);
      size_t RxId = tuner->addArgument(Rx);
      size_t RyId = tuner->addArgument(Ry);
      size_t RzId = tuner->addArgument(Rz);
      size_t stepId = tuner->addArgument(step);


      // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
      tuner->setKernelArguments(kernelId,
          std::vector<size_t>{ iterationId, 
          powerId, tempSrcId, tempDstId,
          grid_colsId, grid_rowsId, borderColsId, borderRowsId,
          CapId, RxId, RyId, RzId, stepId });

      //tuner->enableArgumentPrinting(tempDstId, ofile, ktt::ArgumentPrintCondition::All);
      // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
      tuner->setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

      // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
      tuner->setReferenceClass(kernelId, std::make_unique<referenceHotspot>(tempDstId, grid_rows, grid_cols, pyramid_height, total_iterations, tfile, pfile, borderCols, borderRows, smallBlockCol, smallBlockRow, blockCols, blockRows, grid_height, grid_width, Cap, Rx, Ry, Rz, max_slope, step), std::vector<size_t>{ tempDstId });


    }

    //launchComputation is responsible for actual execution of tuned kernel */
    virtual void launchComputation(const size_t kernelId) override {

        std::vector<ktt::ParameterValue> parameterValues = getCurrentConfiguration();
        auto blocksize = std::get<1>(parameterValues[0]);
        smallBlockCol = blocksize-(pyramid_height)*EXPAND_RATE;
        smallBlockRow = blocksize-(pyramid_height)*EXPAND_RATE;
        blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
        blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
        const ktt::DimensionVector ndRangeDimensions(blocksize*blockCols,blocksize*blockRows, 1);
        const ktt::DimensionVector workGroupDimensions(blocksize, blocksize, 1);
        bool srcPosition = true; //if the original pointer to source temperature is in tempSrc, true. otherwise false.

        int t;
        for (t = 0; t < total_iterations; t += pyramid_height) {
          
          // Specify kernel arguments
          int iter = MIN(pyramid_height, total_iterations - t);
          updateArgumentScalar(iterationId, &iter);
          // Run the kernel
          runKernel(kernelId, ndRangeDimensions, workGroupDimensions);

          if (t + pyramid_height < total_iterations) //if there will be another iteration
          {
            // Swap the source and destination temperatures
            swapKernelArguments(kernelId, tempSrcId, tempDstId);
            srcPosition = !srcPosition;
          }
        }


        // Write final output to output file
        /*if (srcPosition)
          writeoutput(tempDst, grid_rows, grid_cols, ofile);
        else
          writeoutput(tempSrc, grid_rows, grid_cols, ofile);
*/
    }

    void tune() {
        tuner->tuneKernel(kernelId);
        tuner->printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
        tuner->printResult(kernelId, std::string("reduction_output.csv"), ktt::PrintFormat::CSV);
    }

    size_t getKernelId() const {
        return kernelId;
    }
    
  private:
    
    ktt::Tuner* tuner;
    size_t kernelId;
    size_t iterationId;
    size_t tempSrcId;
    size_t tempDstId;


    int size;
    int grid_rows;
    int grid_cols;
    std::vector<float> tempSrc; 
    std::vector<float> tempDst; 
    std::vector<float> power; 
    int pyramid_height; //number of iterations
    int total_iterations;
    int iteration; //TODO comment about why there are various iteration variables


    int blockRows, blockCols;
    int smallBlockRow, smallBlockCol;
    int borderRows, borderCols;
    float grid_height, grid_width;

    float Cap;
    float Rx, Ry, Rz;
    float max_slope;
    float step;

    char* ofile; //the name of output file for temperatures

};
