#pragma once

#include "tuner_api.h"
#include "hotspot.h"

class tunableHotspot : public ktt::TuningManipulator {
  public:

    //Constructor creates internal structures and setups the tuning environment
    // it takes arguments passed from command line arguments and initializes all data variables
    
    tunableHotspot(ktt::Tuner *tuner, size_t kernelId, int grid_rows, int grid_cols, int total_iterations, char *ofile, size_t tempSrcId, size_t tempDstId, size_t iterationId, size_t borderRowsId, size_t borderColsId) : TuningManipulator() {

      this->tuner = tuner;

      // Initialize data variables
      // we need some of them calculated to set the ndRange for example
      this->grid_rows = grid_rows;
      this->grid_cols = grid_cols;
      this->total_iterations = total_iterations;
      this->ofile = ofile;
      
      // Ids of relevant parameters of the kernel, these will be updated
      this->kernelId = kernelId;
      this->tempSrcId = tempSrcId;
      this->tempDstId = tempDstId;
      this->iterationId = iterationId;
      this->borderRowsId = borderRowsId;
      this->borderColsId = borderColsId;

    }

    //launchComputation is responsible for actual execution of tuned kernel */
    virtual void launchComputation(const size_t kernelId) override {

        std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
        int blocksize = parameterValues[0].getValue();
        int pyramid_height = parameterValues[1].getValue();
        int smallBlockCol = blocksize-pyramid_height*EXPAND_RATE;
        int smallBlockRow = blocksize-pyramid_height*EXPAND_RATE;
        int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
        int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
        int borderCols = pyramid_height*EXPAND_RATE/2;
        int borderRows = pyramid_height*EXPAND_RATE/2;
        updateArgumentScalar(borderColsId, &borderCols);
        updateArgumentScalar(borderRowsId, &borderRows);
        
        const ktt::DimensionVector ndRangeDimensions(blocksize*blockCols,blocksize*blockRows, 1);
        const ktt::DimensionVector workGroupDimensions(blocksize, blocksize, 1);
        bool srcPosition = true; //if the original pointer to source temperature is in tempSrc, true. otherwise false.

        int t;
        for (t = 0; t < total_iterations; t += pyramid_height) {
          
          // Specify kernel arguments
          int iter = MIN(pyramid_height, total_iterations - t);
          updateArgumentScalar(iterationId, &iter);
          // run the kernel
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
    size_t borderRowsId;
    size_t borderColsId;

    int grid_rows;
    int grid_cols;
    int total_iterations;
    int iteration; //TODO comment about why there are various iteration variables

    char* ofile; //the name of output file for temperatures

};
