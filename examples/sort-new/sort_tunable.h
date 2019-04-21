#pragma once

#include "tuner_api.h"
static const int SORT_BITS = 32;
typedef unsigned int uint;

static uint nbits = 4;

class TunableSort : public ktt::TuningManipulator {
  public:
    // Constructor takes ids of kernel arguments that will be updated or added
    TunableSort(
        const std::vector<ktt::KernelId>& kernelIds,
        const int size,
        const ktt::ArgumentId keysOutId,
        const ktt::ArgumentId valuesOutId,
        const ktt::ArgumentId keysInId,
        const ktt::ArgumentId valuesInId,
        const ktt::ArgumentId scanNumBlocksId,
        const ktt::ArgumentId countersId,
        const ktt::ArgumentId counterSumsId,
        const ktt::ArgumentId blockOffsetsId,
        const ktt::ArgumentId scanBlocksSumId,
        const ktt::ArgumentId startBitId,
        const ktt::ArgumentId scanOutDataId,
        const ktt::ArgumentId scanInDataId,
        const ktt::ArgumentId scanOneBlockSumId,
        const ktt::ArgumentId numElementsId,
        const ktt::ArgumentId fullBlockId,
        const ktt::ArgumentId storeSumId ):
      kernelIds(kernelIds),
      size(size),
      keysOutId(keysOutId),
      valuesOutId(valuesOutId),
      keysInId(keysInId),
      valuesInId(valuesInId),
      scanNumBlocksId(scanNumBlocksId),
      countersId(countersId),
      counterSumsId(counterSumsId),
      blockOffsetsId(blockOffsetsId),
      scanBlockSumsId(scanBlocksSumId),
      startBitId(startBitId),
      scanOutDataId(scanOutDataId),
      scanInDataId(scanInDataId),
      scanOneBlockSumId(scanOneBlockSumId),
      numElementsId(numElementsId),
      fullBlockId(fullBlockId),
      storeSumId(storeSumId)
    {
      keysOut = std::vector<uint>(size);
      valuesOut = std::vector<uint>(size);
      keysIn = std::vector<uint>(size);
      valuesIn = std::vector<uint>(size);
    }

    // Run the code with kernels
    void launchComputation(const ktt::KernelId) override {

      std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

      int sortBlockSize = (int)getParameterValue("SORT_BLOCK_SIZE", parameterValues);
      int sortVectorSize = (int)getParameterValue("SORT_VECTOR", parameterValues);
      const ktt::DimensionVector workGroupDimensionsSort(sortBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsSort(size/sortVectorSize, 1, 1);

      int scanBlockSize = (int)getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
      int scanVectorSize = (int)getParameterValue("SCAN_VECTOR", parameterValues);
      const ktt::DimensionVector workGroupDimensionsScan(scanBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsScan(size/scanVectorSize, 1, 1);
      uint scanNumBlocks = static_cast<uint>(ndRangeDimensionsScan.getSizeX() / scanBlockSize);
      updateArgumentScalar(scanNumBlocksId, &scanNumBlocks);

      uint countersSize = 16*scanNumBlocks; 
      std::vector<unsigned int> counters(countersSize);
      updateArgumentVector(countersId, counters.data(), countersSize);
      std::vector<unsigned int> counterSums(countersSize);
      updateArgumentVector(counterSumsId, counterSums.data(), countersSize);
      std::vector<unsigned int> blockOffsets(countersSize);
      updateArgumentVector(blockOffsetsId, blockOffsets.data(), countersSize);

      // Allocate space for block sums in the scan kernel.
      uint maxNumScanElements = size;
      uint numScanElts = maxNumScanElements;
      uint level = 0;

      std::vector<std::vector<uint>> scanBlockSums;
      do
      {
        uint numBlocks = std::max(1, (int) ceil((float) numScanElts / (sortVectorSize * scanBlockSize)));
        if (numBlocks > 1) {
          scanBlockSums.push_back(std::vector<uint>(numBlocks));
          level++;
        }
        numScanElts = numBlocks;
      }
      while (numScanElts > 1);
      scanBlockSums.push_back(std::vector<uint>(1));

      uint startbit; 
      bool swap = true;
      for (startbit = 0; startbit < SORT_BITS; startbit += nbits)
      {
        updateArgumentScalar(startBitId, &startbit);

        //radixSortBlocks
        //  <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        //  (nbits, startbit, tempKeys, tempValues, keys, values);
        runKernel(kernelIds[0], ndRangeDimensionsSort, workGroupDimensionsSort);

        //findRadixOffsets
        //  <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        //  ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
        //   findBlocks);
        runKernel(kernelIds[1], ndRangeDimensionsScan, workGroupDimensionsScan);
        getArgumentVector(countersId, counters.data());

        scanArrayRecursive(counterSums, counters, 16*scanNumBlocks, 0, scanBlockSums);

        //reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
        //  (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        //   (uint2*)tempValues, blockOffsets, countersSum, counters,
        //   reorderBlocks);
        updateArgumentVector(counterSumsId, counterSums.data(), counterSums.size());
        updateArgumentVector(countersId, counters.data(), counters.size());
        if (swap) {
          swapKernelArguments(kernelIds[2], keysOutId, keysInId);
          swapKernelArguments(kernelIds[2], valuesOutId, valuesInId);
          swap = !swap;
        }
        runKernel(kernelIds[2], ndRangeDimensionsScan, workGroupDimensionsScan);
      }
    }

void scanArrayRecursive(std::vector<uint> &outArray, std::vector<uint> &inArray, unsigned int numElements, int level, std::vector<std::vector<uint>> &blockSums)
{
    // Kernels handle 8 elems per thread
  std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
  unsigned int scanBlockSize = (unsigned int)getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
  unsigned int sortVectorSize = (unsigned int)getParameterValue("SORT_VECTOR", parameterValues);
  const ktt::DimensionVector workGroupDimensions(scanBlockSize, 1, 1);
  unsigned int numBlocks = std::max(1u, (unsigned int)std::ceil((float)numElements/(sortVectorSize*scanBlockSize)));
  const ktt::DimensionVector ndRangeDimensions(numBlocks*scanBlockSize, 1, 1);


  updateArgumentScalar(numElementsId, &numElements);
  updateArgumentVector(scanOutDataId, outArray.data(), outArray.size());
  updateArgumentVector(scanInDataId, inArray.data(), inArray.size());
  updateArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
  bool fullBlock = (numElements == numBlocks * sortVectorSize * scanBlockSize);
  updateArgumentScalar(fullBlockId, &fullBlock);
  bool storeSum;


  // execute the scan
  if (numBlocks > 1)
  {
    storeSum = 1;

    updateArgumentScalar(storeSumId, &storeSum);
    runKernel(kernelIds[4], ndRangeDimensions, workGroupDimensions); 
    getArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());

    scanArrayRecursive((blockSums[level]), (blockSums[level]),
        numBlocks, level + 1, blockSums);

    //vectorAddUniform4<<< grid, threads >>>
    //  (outArray, blockSums[level], numElements);
    updateArgumentScalar(numElementsId, &numElements);
    updateArgumentVector(scanOutDataId, outArray.data(), outArray.size());
    updateArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
    runKernel(kernelIds[3], ndRangeDimensions, workGroupDimensions);
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());
  } else
  {
    storeSum = 0;
    updateArgumentScalar(storeSumId, &storeSum);
    runKernel(kernelIds[4], ndRangeDimensions, workGroupDimensions);
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());
  }
}

  private:
    std::vector<ktt::KernelId> kernelIds; // Ids of the internal kernels
    int size;
    ktt::ArgumentId keysOutId;
    ktt::ArgumentId valuesOutId;
    std::vector<uint> keysOut;
    std::vector<uint> valuesOut;
    ktt::ArgumentId keysInId;
    ktt::ArgumentId valuesInId;
    std::vector<uint> keysIn;
    std::vector<uint> valuesIn;
    ktt::ArgumentId scanNumBlocksId;
    ktt::ArgumentId countersId;
    ktt::ArgumentId counterSumsId;
    ktt::ArgumentId blockOffsetsId;
    ktt::ArgumentId scanBlockSumsId;
    ktt::ArgumentId startBitId;
    ktt::ArgumentId scanOutDataId;
    std::vector<uint> scanOutData;
    ktt::ArgumentId scanInDataId;
    std::vector<uint> scanInData;
    ktt::ArgumentId scanOneBlockSumId;
    std::vector<uint> scanOneBlockSum;
    ktt::ArgumentId numElementsId;
    ktt::ArgumentId fullBlockId;
    ktt::ArgumentId storeSumId;
};
