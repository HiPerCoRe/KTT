#pragma once

#include "tuner_api.h"
#include "hotspot.h"
#define BLOCK_SIZE_REF 16 //has to be the same value as in reference kernel
#define PYRAMID_HEIGHT_REF 2
#define BLOCK_SIZE_C BLOCK_SIZE_REF
#define BLOCK_SIZE_R BLOCK_SIZE_REF

class referenceManipulator : public ktt::TuningManipulator {
  public:

    //Constructor creates internal structures and setups the tuning environment
    // it takes arguments passed from command line arguments and initializes all data variables
    
    referenceManipulator(ktt::Tuner *tuner, ktt::KernelId referenceKernelId, int grid_rows, int grid_cols, int total_iterations, int size, char *ofile, ktt::ArgumentId tempSrcId, ktt::ArgumentId tempDstId, ktt::ArgumentId iterationId, ktt::ArgumentId borderRowsId, ktt::ArgumentId borderColsId) {

      this->tuner = tuner;

      // Initialize data variables
      // we need some of them calculated to set the ndRange for example
      this->grid_rows = grid_rows;
      this->grid_cols = grid_cols;
      this->total_iterations = total_iterations;
      this->ofile = ofile;
      this->size = size;
      
      // Ids of relevant parameters of the kernel, these will be updated
      this->referenceKernelId = referenceKernelId;
      this->tempSrcId = tempSrcId;
      this->tempDstId = tempDstId;
      this->iterationId = iterationId;
      this->borderRowsId = borderRowsId;
      this->borderColsId = borderColsId;

    }

    //launchComputation is responsible for actual execution of tuned kernel */
    void launchComputation(const ktt::KernelId kernelId) override {


        int smallBlockCol = BLOCK_SIZE_REF-PYRAMID_HEIGHT_REF*EXPAND_RATE;
        int smallBlockRow = BLOCK_SIZE_REF-PYRAMID_HEIGHT_REF*EXPAND_RATE;
        int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
        int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
        int borderCols = PYRAMID_HEIGHT_REF*EXPAND_RATE/2;
        int borderRows = PYRAMID_HEIGHT_REF*EXPAND_RATE/2;
        
        const ktt::DimensionVector ndRangeDimensions(BLOCK_SIZE_REF*blockCols,BLOCK_SIZE_REF*blockRows, 1);
        const ktt::DimensionVector workGroupDimensions(BLOCK_SIZE_REF, BLOCK_SIZE_REF, 1);
        updateArgumentScalar(borderRowsId, &borderRows);
        updateArgumentScalar(borderColsId, &borderCols);
        bool srcPosition = true; //if the original pointer to source temperature is in tempSrc, true. otherwise false.

        int t;
        for (t = 0; t < total_iterations; t += PYRAMID_HEIGHT_REF) {
          
          // Specify kernel arguments
          int iter = MIN(PYRAMID_HEIGHT_REF, total_iterations - t);
          updateArgumentScalar(iterationId, &iter);
          // Run the kernel
          runKernel(kernelId, ndRangeDimensions, workGroupDimensions);

          if (t + PYRAMID_HEIGHT_REF < total_iterations) //if there will be another iteration
          {
            // Swap the source and destination temperatures
            swapKernelArguments(kernelId, tempSrcId, tempDstId);
            srcPosition = !srcPosition;
          }
        }

        if (!srcPosition)
        {
          copyArgumentVector(tempDstId, tempSrcId, size);
        }
    }

    ktt::KernelId getKernelId() const {
        return referenceKernelId;
    }
    
  private:
    
    ktt::Tuner* tuner;
    ktt::KernelId referenceKernelId; //the id of this kernel
    ktt::ArgumentId iterationId;
    ktt::ArgumentId tempSrcId;
    ktt::ArgumentId tempDstId;
    ktt::ArgumentId borderRowsId;
    ktt::ArgumentId borderColsId;

    int grid_rows;
    int grid_cols;
    int total_iterations;
    int iteration; //TODO comment about why there are various iteration variables
    int size;

    char* ofile; //the name of output file for temperatures

};
