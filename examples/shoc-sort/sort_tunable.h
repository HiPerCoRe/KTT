#pragma once

#include "tuner_api.h"

class tunableSort : public ktt::TuningManipulator {
  public:
    // Constructor creates internal structures and setups the environment
    // it takes arguments from command line and generated input data
    tunableSort(ktt::Tuner *tuner, std::vector<size_t> kernelIds, int size, size_t inId, size_t outId, size_t isumsId, size_t sizeId, size_t localMem1Id, size_t localMem2Id, size_t workGroupSizeId, size_t shiftId) : TuningManipulator() 
  {
    this->tuner = tuner;

    // ids of kernels' agruments that will be updated or added
    this->kernelIds = kernelIds;
    this->size = size;
    this->inId = inId;
    this->outId = outId;
    this->isumsId = isumsId;
    this->sizeId = sizeId;
    this->localMem1Id = localMem1Id;
    this->localMem2Id = localMem2Id;
    this->workGroupSizeId = workGroupSizeId;
    this->shiftId = shiftId;
    kernelId = -1;

  }

    //run the code with kernels
    void launchComputation(const size_t kernelId) override {
      const ktt::DimensionVector ndRangeDimensions(16384, 1, 1);
      const ktt::DimensionVector workGroupDimensions(256, 1, 1);

      size_t localSize = workGroupDimensions.getSizeX();
      //updateArgumentLocal(localMem1Id, localSize);
      //updateArgumentLocal(localMem2Id, 2 * localSize);
      updateArgumentScalar(workGroupSizeId, &localSize);
      
      for (int shift = 0; shift < sizeof(unsigned int)*8; shift += radix_width)
      {
        // Like scan, we use a reduce-then-scan approach

        // But before proceeding, update the shift appropriately
        // for each kernel. This is how many bits to shift to the
        // right used in binning.
        updateArgumentScalar(shiftId, &shift);

        // Also, the sort is not in place, so swap the input and output
        // buffers on each pass.
        bool even = ((shift / radix_width) % 2 == 0) ? true : false;

        if (even)
        {
          changeKernelArguments(kernelIds[0], {inId, isumsId, 3 /* == sizeId*/, localMem1Id, shiftId});
        }
        else // i.e. odd pass
        {
          changeKernelArguments(kernelIds[0], {outId, isumsId, sizeId, localMem1Id, shiftId});
        }

        // Each thread block gets an equal portion of the
        // input array, and computes occurrences of each digit.
        runKernel(kernelIds[0], ndRangeDimensions, workGroupDimensions);
        printf("Kenrel reduce done\n------------------------------------------------------------\n");

        // Next, a top-level exclusive scan is performed on the
        // per block histograms.  This is done by a single
        // work group (note global size here is the same as local).
        runKernel(kernelIds[1], workGroupDimensions, workGroupDimensions);
        printf("Kernel top scan done\n------------------------------------------------------------\n");

        // Finally, a bottom-level scan is performed by each block
        // that is seeded with the scanned histograms which rebins,
        // locally scans, then scatters keys to global memory
        runKernel(kernelIds[2], ndRangeDimensions, workGroupDimensions);
        printf("Kernel bottom scan done\n------------------------------------------------------------\n");
        
        // Also, the sort is not in place, so swap the input and output
        // buffers on each pass.
        swapKernelArguments(kernelIds[2], inId, outId);
      }

    }

    void tune() {
      tuner->tuneKernel(kernelId);
      tuner->printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
      tuner->printResult(kernelId, std::string("sort_result.csv"), ktt::PrintFormat::CSV);
    }

    void setKernelId(size_t id) {
      kernelId = id;
    }

  private:

    ktt::Tuner* tuner;
    size_t inId;
    size_t outId;
    size_t shiftId;
    size_t isumsId;
    size_t sizeId;
    size_t localMem1Id;
    size_t localMem2Id;
    size_t workGroupSizeId;
    size_t kernelId; //id of the composite kernel
    std::vector<size_t> kernelIds; //ids of the internal kernels

    int shift;
    int size;
};
