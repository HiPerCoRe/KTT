#pragma once

#include "tuner_api.h"

class tunableSort : public ktt::TuningManipulator {
  public:
    // Constructor creates internal structures and setups the environment
    // it takes arguments from command line and generated input data
    tunableSort(ktt::Tuner *tuner, std::vector<size_t> kernelIds, int size, size_t inId, size_t outId, size_t isumsId, size_t sizeId, size_t localMem1Id, size_t localMem2Id, size_t localMem3Id, size_t numberOfGroupsId, size_t shiftId) : TuningManipulator() 
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
    this->localMem3Id = localMem3Id;
    this->numberOfGroupsId = numberOfGroupsId;
    this->shiftId = shiftId;
    kernelId = -1;

  }

    //run the code with kernels
    virtual void launchComputation(const size_t kernelId) override {

      std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
      int localSize = (int)parameterValues[0].getValue();
      const ktt::DimensionVector workGroupDimensions(localSize, 1, 1);
      int globalSize = (int)parameterValues[1].getValue();
      const ktt::DimensionVector ndRangeDimensions(globalSize, 1, 1);
      
      int numberOfGroups = globalSize/localSize;
      updateArgumentScalar(numberOfGroupsId, &numberOfGroups);
      updateArgumentLocal(localMem1Id, localSize); //local, workgroupsize*sizeof(unsigned int)
      updateArgumentLocal(localMem2Id, 2*localSize);
      updateArgumentLocal(localMem3Id, 2*localSize);//local, 2*workgroupsize*sizeof(unsigned int)
      int isumsSize = (numberOfGroups*16*sizeof(unsigned int));
      unsigned int* is = new unsigned int[isumsSize];
      updateArgumentVector(isumsId, is, isumsSize);
//vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups

      bool inOutSwapped = false;
      unsigned int *out = new unsigned int[size];
      unsigned int *in = new unsigned int[size];

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

        // Next, a top-level exclusive scan is performed on the
        // per block histograms.  This is done by a single
        // work group (note global size here is the same as local).
        runKernel(kernelIds[1], workGroupDimensions, workGroupDimensions);

        // Finally, a bottom-level scan is performed by each block
        // that is seeded with the scanned histograms which rebins,
        // locally scans, then scatters keys to global memory
        runKernel(kernelIds[2], ndRangeDimensions, workGroupDimensions);
        
        // Also, the sort is not in place, so swap the input and output
        // buffers on each pass.
        swapKernelArguments(kernelIds[2], inId, outId);
        if (shift+radix_width < sizeof(unsigned int)*8) //this is not the last iteration
          inOutSwapped = !inOutSwapped;

      }
      if (inOutSwapped) { //copy contents of in to out, since they are swapped
        getArgumentVector(outId, out);
        getArgumentVector(inId, in);
        std::copy(in, in+size, out);
        updateArgumentVector(outId, out);
      }
      delete[] in;
      delete[] out;
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
    size_t localMem3Id;
    size_t numberOfGroupsId;
    size_t kernelId; //id of the composite kernel
    std::vector<size_t> kernelIds; //ids of the internal kernels

    int shift;
    int size;
};
