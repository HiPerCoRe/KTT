#pragma once

#include "tuner_api.h"

class TunableSort : public ktt::TuningManipulator {
  public:
    // Constructor takes ids of kernel arguments that will be updated or added
    TunableSort(const std::vector<ktt::KernelId>& kernelIds, const int size, const ktt::ArgumentId inId, const ktt::ArgumentId outId,
      const ktt::ArgumentId isumsId, const ktt::ArgumentId sizeId, const ktt::ArgumentId localMem1Id, const ktt::ArgumentId localMem2Id,
      const ktt::ArgumentId localMem3Id, const ktt::ArgumentId numberOfGroupsId, const ktt::ArgumentId shiftId) :
      kernelIds(kernelIds),
      size(size),
      inId(inId),
      outId(outId),
      isumsId(isumsId),
      sizeId(sizeId),
      localMem1Id(localMem1Id),
      localMem2Id(localMem2Id),
      localMem3Id(localMem3Id),
      numberOfGroupsId(numberOfGroupsId),
      shiftId(shiftId)
    {}

    // Run the code with kernels
    void launchComputation(const ktt::KernelId) override {

      const int radix_width = 4;
      std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
      size_t localSize = getParameterValue("LOCAL_SIZE", parameterValues);
      size_t globalSize = getParameterValue("GLOBAL_SIZE", parameterValues);
      
      int numberOfGroups = static_cast<int>(globalSize / localSize);
      updateArgumentScalar(numberOfGroupsId, &numberOfGroups);
      updateArgumentLocal(localMem1Id, localSize); // Local, workgroupsize * sizeof(unsigned int)
      updateArgumentLocal(localMem2Id, 2 * localSize);
      updateArgumentLocal(localMem3Id, 2 * localSize); // Local, 2 * workgroupsize * sizeof(unsigned int)
      int isumsSize = (numberOfGroups * 16 * sizeof(unsigned int));
      std::vector<unsigned int> is(isumsSize);
      // Vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups
      updateArgumentVector(isumsId, is.data(), isumsSize);

      bool inOutSwapped = false;

      for (int shift = 0; shift < sizeof(unsigned int) * 8; shift += radix_width)
      {
        // Like scan, we use a reduce-then-scan approach

        // But before proceeding, update the shift appropriately for each kernel. This is how many bits to shift to the right used in binning.
        updateArgumentScalar(shiftId, &shift);

        // Also, the sort is not in place, so swap the input and output buffers on each pass.
        bool even = ((shift / radix_width) % 2 == 0) ? true : false;

        if (even)
        {
          changeKernelArguments(kernelIds[0], {inId, isumsId, sizeId, localMem1Id, shiftId});
        }
        else
        {
          changeKernelArguments(kernelIds[0], {outId, isumsId, sizeId, localMem1Id, shiftId});
        }

        // Each thread block gets an equal portion of the input array, and computes occurrences of each digit.
        runKernel(kernelIds[0]);

        // Next, a top-level exclusive scan is performed on the per block histograms. This is done by a single work group (note global size here
        // is the same as local).
        runKernel(kernelIds[1]);

        // Finally, a bottom-level scan is performed by each block that is seeded with the scanned histograms which rebins, locally scans, then
        // scatters keys to global memory
        runKernel(kernelIds[2]);
        
        // Also, the sort is not in place, so swap the input and output buffers on each pass.
        swapKernelArguments(kernelIds[2], inId, outId);
        if (shift + radix_width < sizeof(unsigned int) * 8) // Not the last iteration
          inOutSwapped = !inOutSwapped;

      }
      if (inOutSwapped) { // Copy contents of in to out, since they are swapped
        copyArgumentVector(outId, inId, size);
      }
    }

  private:
    std::vector<ktt::KernelId> kernelIds; // Ids of the internal kernels
    int size;
    ktt::ArgumentId inId;
    ktt::ArgumentId outId;
    ktt::ArgumentId isumsId;
    ktt::ArgumentId sizeId;
    ktt::ArgumentId localMem1Id;
    ktt::ArgumentId localMem2Id;
    ktt::ArgumentId localMem3Id;
    ktt::ArgumentId numberOfGroupsId;
    ktt::ArgumentId shiftId;
};
