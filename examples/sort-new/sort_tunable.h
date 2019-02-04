#pragma once

#include "tuner_api.h"
static const int SORT_BITS = 32;
typedef unsigned int uint;

static uint nbits = 4;

class TunableSort : public ktt::TuningManipulator {
  public:
    // Constructor takes ids of kernel arguments that will be updated or added
    TunableSort(const std::vector<ktt::KernelId>& kernelIds, const ktt::DeviceInfo device, const int size, const ktt::ArgumentId keysOutId, const ktt::ArgumentId valuesOutId, const ktt::ArgumentId keysInId, const ktt::ArgumentId valuesInId, std::vector<uint> & keysIn, std::vector<uint> & valuesIn, const ktt::ArgumentId sortNumBlocksId, const ktt::ArgumentId sortBlockSizeId, const ktt::ArgumentId scanNumBlocksId, const ktt::ArgumentId scanBlockSizeId, const ktt::ArgumentId countersId, const ktt::ArgumentId counterSumsId, const ktt::ArgumentId blockOffsetsId, const ktt::ArgumentId scanBlocksSumId, const ktt::ArgumentId startBitId, const ktt::ArgumentId scanOutDataId, const ktt::ArgumentId scanInDataId, const ktt::ArgumentId scanOneBlockSumId, const ktt::ArgumentId numElementsId, const ktt::ArgumentId fullBlockId, const ktt::ArgumentId storeSumId, const ktt::ArgumentId vectorOutDataId, const ktt::ArgumentId vectorBlockSumId, const ktt::ArgumentId vectorNumElementsId) :
      kernelIds(kernelIds),
      device(device),
      size(size),
      keysOutId(keysOutId),
      valuesOutId(valuesOutId),
      keysInId(keysInId),
      valuesInId(valuesInId),
      keysIn(keysIn),
      valuesIn(valuesIn),
      sortNumBlocksId(sortNumBlocksId),
      sortBlockSizeId(sortBlockSizeId),
      scanNumBlocksId(scanNumBlocksId),
      scanBlockSizeId(scanBlockSizeId),
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
      storeSumId(storeSumId),
      vectorOutDataId(vectorOutDataId),
      vectorBlockSumId(vectorBlockSumId),
      vectorNumElementsId(vectorNumElementsId)
    {
      keysOut = std::vector<uint>(size);
      valuesOut = std::vector<uint>(size);
    }

    // Run the code with kernels
    void launchComputation(const ktt::KernelId) override {

      printf( "launchComputation START\n");
      std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

      int sortBlockSize = (int)getParameterValue("SORT_BLOCK_SIZE", parameterValues);
      const ktt::DimensionVector workGroupDimensionsSort(sortBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsSort(size/4, 1, 1);
      uint sortNumBlocks = ndRangeDimensionsSort.getSizeX() / sortBlockSize;
      //updateArgumentScalar(sortNumBlocksId, &sortNumBlocks);
      //updateArgumentScalar(sortBlockSizeId, &sortBlockSize);
      
      int scanBlockSize = (int)getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
      const ktt::DimensionVector workGroupDimensionsScan(scanBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsScan(size/2, 1, 1);
      uint scanNumBlocks = ndRangeDimensionsScan.getSizeX() / scanBlockSize;
      updateArgumentScalar(scanNumBlocksId, &scanNumBlocks);
      //updateArgumentScalar(scanBlockSizeId, &scanBlockSize);

      //int warpSize = (int)getParameterValue("WARP_SIZE", parameterValues);
      uint countersSize = 16*scanNumBlocks; //warpSize * sortNumBlocks; 
      std::vector<unsigned int> counters(countersSize);
      updateArgumentVector(countersId, counters.data(), countersSize);
      std::vector<unsigned int> counterSums(countersSize);
      updateArgumentVector(counterSumsId, counterSums.data(), countersSize);
      std::vector<unsigned int> blockOffsets(countersSize);
      updateArgumentVector(blockOffsetsId, blockOffsets.data(), countersSize);

      // Allocate space for block sums in the scan kernel.
      uint numLevelsAllocated = 0;
      uint maxNumScanElements = size;
      uint numScanElts = maxNumScanElements;
      uint level = 0;

      std::vector<std::vector<uint>> scanBlockSums;
      do
      {
        uint numBlocks = std::max(1, (int) ceil((float) numScanElts / (4 * scanBlockSize)));
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

        //radixSortBlocks
        //  <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        //  (nbits, startbit, tempKeys, tempValues, keys, values);
        printf( "KERNEL 0 start\n");
        updateArgumentScalar(startBitId, &startbit);
        updateArgumentVector(keysOutId, keysOut.data());
        updateArgumentVector(valuesOutId, valuesOut.data());
        updateArgumentVector(keysInId, keysIn.data());
        updateArgumentVector(valuesInId, valuesIn.data());
        runKernel(kernelIds[0], ndRangeDimensionsSort, workGroupDimensionsSort);
        getArgumentVector(keysOutId, keysOut.data());
        getArgumentVector(valuesOutId, valuesOut.data());
        printf( "KERNEL 0 end\n");
        
        //findRadixOffsets
        //  <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        //  ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
        //   findBlocks);
        printf( "KERNEL 1 start\n");
        updateArgumentVector(keysOutId, keysOut.data());
        updateArgumentVector(countersId, counters.data());
        updateArgumentVector(blockOffsetsId, blockOffsets.data());
        runKernel(kernelIds[1], ndRangeDimensionsScan, workGroupDimensionsScan);
        getArgumentVector(countersId, counters.data());
        getArgumentVector(blockOffsetsId, blockOffsets.data());
       // getArgumentVector(keysOutId, keysOut.data());
        printf( "KERNEL 1 end\n");

        printf( "scanArrayRecursive start\n");
        scanArrayRecursive(counterSums, counters, 16*scanNumBlocks, 0, scanBlockSums);
        printf( "scanArrayRecursive end\n");

        //reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
        //  (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        //   (uint2*)tempValues, blockOffsets, countersSum, counters,
        //   reorderBlocks);
        printf( "KERNEL 2 start\n");
        updateArgumentVector(counterSumsId, counterSums.data(), counterSums.size());
        updateArgumentVector(countersId, counters.data(), counters.size());
        updateArgumentVector(blockOffsetsId, blockOffsets.data(), blockOffsets.size());
        updateArgumentVector(keysOutId, keysOut.data());
        updateArgumentVector(valuesOutId, valuesOut.data());
        if (swap) {
          swapKernelArguments(kernelIds[2], keysOutId, keysInId);
          swapKernelArguments(kernelIds[2], valuesOutId, valuesInId);
          swap = !swap;
        }
        runKernel(kernelIds[2], ndRangeDimensionsScan, workGroupDimensionsScan);
        getArgumentVector(keysOutId, keysOut.data());
        getArgumentVector(valuesOutId, valuesOut.data());
        getArgumentVector(keysInId, keysIn.data());
        getArgumentVector(valuesInId, valuesIn.data());
        printf( "KERNEL 2 end\n");

      }
      if (!swap) {
          updateArgumentVector(keysOutId, keysIn.data());
          updateArgumentVector(valuesOutId, valuesIn.data());
      }
    }

void scanArrayRecursive(std::vector<uint> &outArray, std::vector<uint> &inArray, int numElements, int level, std::vector<std::vector<uint>> &blockSums)
{
    // Kernels handle 8 elems per thread
  std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
  int scanBlockSize = (int)getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
  const ktt::DimensionVector workGroupDimensions(scanBlockSize, 1, 1);
  unsigned int numBlocks = std::max(1u, (unsigned int)std::ceil((float)numElements/(4.f*scanBlockSize)));
  const ktt::DimensionVector ndRangeDimensions(numBlocks*scanBlockSize, 1, 1);

  printf( "scanArrayRecursive start inside %u\n", numElements);
    fflush(stdout);

  updateArgumentScalar(numElementsId, &numElements);
  updateArgumentVector(scanOutDataId, outArray.data(), outArray.size());
  updateArgumentVector(scanInDataId, inArray.data(), inArray.size());
  updateArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
  bool fullBlock = (numElements == numBlocks * 4 * scanBlockSize);
  updateArgumentScalar(fullBlockId, &fullBlock);
  bool storeSum;


  // execute the scan
  if (numBlocks > 1)
  {
    storeSum = 1;
    printf( "scanArrayRecursive inside numBlocks > 1\n");
    fflush(stdout);

    updateArgumentScalar(storeSumId, &storeSum);
    runKernel(kernelIds[4], ndRangeDimensions, workGroupDimensions); 
    getArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());
    printf( "scanArrayRecursive end inside numBlocks > 1\n");
    fflush(stdout);
    printf( "scanArrayRecursive calling resursive\n");
    fflush(stdout);
    scanArrayRecursive((blockSums[level]), (blockSums[level]),
        numBlocks, level + 1, blockSums);
    printf( "scanArrayRecursive end calling resursive\n");
    fflush(stdout);
    //vectorAddUniform4<<< grid, threads >>>
    //  (outArray, blockSums[level], numElements);
    uint vectorNumElements = numElements;
    updateArgumentScalar(numElementsId, &numElements);
    updateArgumentVector(scanOutDataId, outArray.data(), outArray.size());
    updateArgumentVector(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size());
    printf( "KERNEL 3 start\n");
    fflush(stdout);
    runKernel(kernelIds[3], ndRangeDimensions, workGroupDimensions);
    printf("KERNEL 3 end\n");
    fflush(stdout);
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());
  } else
  {
    storeSum = 0;
    printf( "scanArrayRecursive inside numBlocks <= 1\n");
    fflush(stdout);
    updateArgumentScalar(storeSumId, &storeSum);
    runKernel(kernelIds[4], ndRangeDimensions, workGroupDimensions);
    getArgumentVector(scanOutDataId, outArray.data(), outArray.size());
    printf( "scanArrayRecursive  end inside numBlocks <= 1\n");
    fflush(stdout);
  }
}

  private:
    std::vector<ktt::KernelId> kernelIds; // Ids of the internal kernels
    ktt::DeviceInfo device;
    int size;
    ktt::ArgumentId keysOutId;
    std::vector<uint> keysOut;
    ktt::ArgumentId valuesOutId;
    std::vector<uint> valuesOut;
    ktt::ArgumentId keysInId;
    std::vector<uint> keysIn;
    ktt::ArgumentId valuesInId;
    std::vector<uint> valuesIn;
    ktt::ArgumentId sortNumBlocksId;
    ktt::ArgumentId sortBlockSizeId;
    ktt::ArgumentId scanNumBlocksId;
    ktt::ArgumentId scanBlockSizeId;
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
    ktt::ArgumentId vectorOutDataId;
    ktt::ArgumentId vectorBlockSumId;
    ktt::ArgumentId vectorNumElementsId;
};
