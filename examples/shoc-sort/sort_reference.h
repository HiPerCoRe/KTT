#pragma once

#include "tuner_api.h"

//the main code and reference kernel adopted from SHOC benchmark, example sort
class referenceSort : public ktt::TuningManipulator {
  public:
    // Constructor creates internal structures and setups the environment
    // it takes arguments from command line and generated input data
    referenceSort(ktt::Tuner *tuner, std::vector<size_t> kernelIds, int size, size_t inId, size_t outId, size_t isumsId, size_t localMem1Id, size_t localMem2Id, size_t workGroupSizeId, size_t shiftId) : TuningManipulator() 
  {
    this->tuner = tuner;

    // ids of kernels' agruments that will be updated or added
    this->kernelIds = kernelIds;
    this->size = size;
    this->inId = inId;
    this->outId = outId;
    this->isumsId = isumsId;
    this->localMem1Id = localMem1Id;
    this->localMem2Id = localMem2Id;
    this->workGroupSizeId = workGroupSizeId;
    this->shiftId = shiftId;

  }

    //run the code with kernels
    virtual void launchComputation(const size_t kernelId) override {
      const ktt::DimensionVector ndRangeDimensions(16384, 1, 1);
      const ktt::DimensionVector workGroupDimensions(256, 1, 1);

      int localSize = workGroupDimensions.getSizeX();
      updateArgumentLocal(localMem1Id, localSize * sizeof(unsigned int));
      updateArgumentLocal(localMem2Id, 2 * localSize * sizeof(unsigned int));
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
        swapKernelArguments(kernelIds[0], inId, outId);
        swapKernelArguments(kernelIds[2], inId, outId);

        // Each thread block gets an equal portion of the
        // input array, and computes occurrences of each digit.
        runKernel(kernelIds[0], ndRangeDimensions, workGroupDimensions);

        // Next, a top-level exclusive scan is performed on the
        // per block histograms.  This is done by a single
        // work group (note global size here is the same as local).
        runKernel(kernelIds[1], workGroupDimensions, workGroupDimensions);

        // Finally, a bottom-level scan is performed by each block
        // that is seeded with the scanned histograms which rebins,
        // locally scans, then scatters keys to global memory
        runKernel(kernelIds[2], ndRangeDimensions, workGroupDimensions);
      }

    }

  private:

    ktt::Tuner* tuner;
    size_t inId;
    size_t outId;
    size_t shiftId;
    size_t isumsId;
    size_t localMem1Id;
    size_t localMem2Id;
    size_t workGroupSizeId;
    std::vector<size_t> kernelIds;

    int shift;
    int size;
};
