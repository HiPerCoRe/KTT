#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Sort2/Sort2.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Sort2/Sort2.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

void ScanArrayRecursive(ktt::ComputeInterface& interface, const std::vector<ktt::KernelDefinitionId>& definitionIds,
    const ktt::ArgumentId numElementsId, const ktt::ArgumentId fullBlockId, const ktt::ArgumentId storeSumId,
    const ktt::ArgumentId scanInDataId, const ktt::ArgumentId scanOutDataId, const ktt::ArgumentId scanOneBlockSumId,
    std::vector<unsigned int>& outArray, std::vector<unsigned int>& inArray, unsigned int numElements, int level,
    std::vector<std::vector<unsigned int>>& blockSums)
{
    // Kernels handle 8 elems per thread
    const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
    unsigned int scanBlockSize = (unsigned int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_BLOCK_SIZE");
    unsigned int sortVectorSize = (unsigned int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_VECTOR");
    const ktt::DimensionVector workGroupDimensions(scanBlockSize, 1, 1);
    unsigned int numBlocks = std::max(1u, (unsigned int)std::ceil((float)numElements/(sortVectorSize*scanBlockSize)));
    const ktt::DimensionVector ndRangeDimensions(numBlocks*scanBlockSize, 1, 1);

    interface.UpdateScalarArgument(numElementsId, &numElements);
    interface.UpdateBuffer(scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
    interface.UpdateBuffer(scanInDataId, inArray.data(), inArray.size() * sizeof(unsigned int));
    interface.UpdateBuffer(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
    bool fullBlock = (numElements == numBlocks * sortVectorSize * scanBlockSize);
    interface.UpdateScalarArgument(fullBlockId, &fullBlock);
    bool storeSum;

    // execute the scan
    if (numBlocks > 1)
    {
        storeSum = 1;

        interface.UpdateScalarArgument(storeSumId, &storeSum);
        interface.RunKernel(definitionIds[4], ndRangeDimensions, workGroupDimensions); 
        interface.DownloadBuffer(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
        interface.DownloadBuffer(scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));

        ScanArrayRecursive(interface, definitionIds, numElementsId, fullBlockId, storeSumId, scanInDataId, scanOutDataId,
            scanOneBlockSumId, (blockSums[level]), (blockSums[level]), numBlocks, level + 1, blockSums);

        interface.UpdateScalarArgument(numElementsId, &numElements);
        interface.UpdateBuffer(scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
        interface.UpdateBuffer(scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
        interface.RunKernel(definitionIds[3], ndRangeDimensions, workGroupDimensions);
        interface.DownloadBuffer(scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
    }
    else
    {
        storeSum = 0;
        interface.UpdateScalarArgument(storeSumId, &storeSum);
        interface.RunKernel(definitionIds[4], ndRangeDimensions, workGroupDimensions);
        interface.DownloadBuffer(scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
    }
}

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;
    int problemSize = 32; // In MiB

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
            }
        }
    }

    if (argc >= 5)
    {
        problemSize = atoi(argv[4]);
    }
  
    int size = problemSize * 1024 * 1024 / sizeof(unsigned int);
    const unsigned int SORT_BITS = 32;
    const unsigned int nbits = 4;

    // Create input and output vectors and initialize with pseudorandom numbers
    std::vector<unsigned int> keysIn(size);
    std::vector<unsigned int> keysOut(size);
    std::vector<unsigned int> valuesIn(size);
    std::vector<unsigned int> valuesOut(size);

    srand(123);

    for (int i = 0; i < size; ++i)
    {
        valuesIn[i] = keysIn[i] = i % 1024;
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Declare kernels and their dimensions
    std::vector<ktt::KernelDefinitionId> definitionIds(5);
    const ktt::DimensionVector ndRangeDimensions;
    const ktt::DimensionVector workGroupDimensions;

    definitionIds[0] = tuner.AddKernelDefinitionFromFile("radixSortBlocks", kernelFile, ndRangeDimensions, workGroupDimensions);
    definitionIds[1] = tuner.AddKernelDefinitionFromFile("findRadixOffsets", kernelFile, ndRangeDimensions, workGroupDimensions);
    definitionIds[2] = tuner.AddKernelDefinitionFromFile("reorderData", kernelFile, ndRangeDimensions, workGroupDimensions);
    definitionIds[3] = tuner.AddKernelDefinitionFromFile("vectorAddUniform4", kernelFile, ndRangeDimensions, workGroupDimensions);
    definitionIds[4] = tuner.AddKernelDefinitionFromFile("scan", kernelFile, ndRangeDimensions, workGroupDimensions);

    // Add arguments for kernels
    // All parameters with foo values (empty vectors or scalar 1) will be updated in tuning manipulator, as their value depends on tuning parameters
    const ktt::ArgumentId nbitsId = tuner.AddArgumentScalar(nbits);
    const ktt::ArgumentId startBitId = tuner.AddArgumentScalar(0);
    const ktt::ArgumentId sizeId = tuner.AddArgumentScalar(size);
  
    const ktt::ArgumentId keysOutId = tuner.AddArgumentVector(keysOut, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId valuesOutId = tuner.AddArgumentVector(valuesOut, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId keysInId = tuner.AddArgumentVector(keysIn, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId valuesInId = tuner.AddArgumentVector(valuesIn, ktt::ArgumentAccessType::ReadWrite);

    const ktt::ArgumentId countersId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId counterSumsId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId blockOffsetsId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
  
    const ktt::ArgumentId scanNumBlocksId = tuner.AddArgumentScalar(1);
    const ktt::ArgumentId numElementsId = tuner.AddArgumentScalar(1);

    const ktt::ArgumentId scanOutDataId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId scanInDataId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId scanOneBlockSumId = tuner.AddArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId fullBlockId = tuner.AddArgumentScalar(1);
    const ktt::ArgumentId storeSumId = tuner.AddArgumentScalar(1);

    const ktt::KernelId kernel = tuner.CreateCompositeKernel("Sort", definitionIds, [&definitionIds, size, scanNumBlocksId, countersId,
        counterSumsId, blockOffsetsId, startBitId, keysInId, keysOutId, valuesInId, valuesOutId, numElementsId, fullBlockId, storeSumId,
        scanInDataId, scanOutDataId, scanOneBlockSumId, SORT_BITS, nbits](ktt::ComputeInterface& interface)
    {
        const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();

        int sortBlockSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_BLOCK_SIZE");
        int sortVectorSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_VECTOR");
        const ktt::DimensionVector workGroupDimensionsSort(sortBlockSize, 1, 1);
        const ktt::DimensionVector ndRangeDimensionsSort(size / sortVectorSize, 1, 1);

        int scanBlockSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_BLOCK_SIZE");
        int scanVectorSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_VECTOR");
        const ktt::DimensionVector workGroupDimensionsScan(scanBlockSize, 1, 1);
        const ktt::DimensionVector ndRangeDimensionsScan(size / scanVectorSize, 1, 1);

        unsigned int scanNumBlocks = static_cast<unsigned int>(ndRangeDimensionsScan.GetSizeX() / scanBlockSize);
        interface.UpdateScalarArgument(scanNumBlocksId, &scanNumBlocks);

        unsigned int countersSize = 16 * scanNumBlocks;
        std::vector<unsigned int> counters(countersSize);
        interface.ResizeBuffer(countersId, countersSize * sizeof(unsigned int), false);
        interface.ResizeBuffer(scanInDataId, countersSize * sizeof(unsigned int), false);
        interface.UpdateBuffer(countersId, counters.data());

        std::vector<unsigned int> counterSums(countersSize);
        interface.ResizeBuffer(counterSumsId, countersSize * sizeof(unsigned int), false);
        interface.ResizeBuffer(scanOutDataId, countersSize * sizeof(unsigned int), false);
        interface.UpdateBuffer(counterSumsId, counterSums.data());

        std::vector<unsigned int> blockOffsets(countersSize);
        interface.ResizeBuffer(blockOffsetsId, countersSize * sizeof(unsigned int), false);
        interface.UpdateBuffer(blockOffsetsId, blockOffsets.data());

        // Allocate space for block sums in the scan kernel.
        unsigned int maxNumScanElements = size;
        unsigned int numScanElts = maxNumScanElements;
        unsigned int level = 0;

        std::vector<std::vector<unsigned int>> scanBlockSums;

        do
        {
            unsigned int numBlocks = std::max(1, (int)ceil((float)numScanElts / (sortVectorSize * scanBlockSize)));
            
            if (numBlocks > 1)
            {
                scanBlockSums.push_back(std::vector<unsigned int>(numBlocks));
                ++level;
            }

            numScanElts = numBlocks;
        }
        while (numScanElts > 1);

        scanBlockSums.push_back(std::vector<unsigned int>(1));
        interface.ResizeBuffer(scanOneBlockSumId, scanBlockSums[0].size() * sizeof(unsigned int), false);

        unsigned int startbit;
        bool swap = true;

        for (startbit = 0; startbit < SORT_BITS; startbit += nbits)
        {
            interface.UpdateScalarArgument(startBitId, &startbit);

            //radixSortBlocks
            //  <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
            //  (nbits, startbit, tempKeys, tempValues, keys, values);
            interface.RunKernel(definitionIds[0], ndRangeDimensionsSort, workGroupDimensionsSort);

            //findRadixOffsets
            //  <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
            //  ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
            //   findBlocks);
            interface.RunKernel(definitionIds[1], ndRangeDimensionsScan, workGroupDimensionsScan);
            interface.DownloadBuffer(countersId, counters.data());

            ScanArrayRecursive(interface, definitionIds, numElementsId, fullBlockId, storeSumId, scanInDataId, scanOutDataId,
                scanOneBlockSumId, counterSums, counters, 16 * scanNumBlocks, 0, scanBlockSums);

            //reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
            //  (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
            //   (uint2*)tempValues, blockOffsets, countersSum, counters,
            //   reorderBlocks);
            interface.UpdateBuffer(counterSumsId, counterSums.data(), counterSums.size() * sizeof(unsigned int));
            interface.UpdateBuffer(countersId, counters.data(), counters.size() * sizeof(unsigned int));

            if (swap)
            {
                interface.SwapArguments(definitionIds[2], keysOutId, keysInId);
                interface.SwapArguments(definitionIds[2], valuesOutId, valuesInId);
                swap = !swap;
            }

            interface.RunKernel(definitionIds[2], ndRangeDimensionsScan, workGroupDimensionsScan);
        }
    });

    //radixSortBlocks
    tuner.SetArguments(definitionIds[0], {nbitsId, startBitId, keysOutId, valuesOutId, keysInId, valuesInId});

    //findRadixOffsets
    tuner.SetArguments(definitionIds[1], {keysOutId, countersId, blockOffsetsId, startBitId, sizeId, scanNumBlocksId});

    //reorderData
    tuner.SetArguments(definitionIds[2], {startBitId, keysOutId, valuesOutId, keysInId, valuesInId, blockOffsetsId, counterSumsId, countersId, scanNumBlocksId});

    //vectorAddUniform
    tuner.SetArguments(definitionIds[3], {scanOutDataId, scanOneBlockSumId, numElementsId});

    //scan
    tuner.SetArguments(definitionIds[4], {scanOutDataId, scanInDataId, scanOneBlockSumId, numElementsId, fullBlockId, storeSumId});

    tuner.AddParameter(kernel, "SORT_BLOCK_SIZE", std::vector<uint64_t>{32, 64, 128, 256, 512, 1024});
    tuner.AddParameter(kernel, "SCAN_BLOCK_SIZE", std::vector<uint64_t>{32, 64, 128, 256, 512, 1024});
    tuner.AddParameter(kernel, "SORT_VECTOR", std::vector<uint64_t>{2, 4, 8});
    tuner.AddParameter(kernel, "SCAN_VECTOR", std::vector<uint64_t>{2, 4, 8});
    auto workGroupConstraint = [](const std::vector<uint64_t>& vector) {return (float)vector.at(1) / vector.at(0) == (float)vector.at(2) / vector.at(3);};
    tuner.AddConstraint(kernel, {"SORT_BLOCK_SIZE", "SCAN_BLOCK_SIZE", "SORT_VECTOR", "SCAN_VECTOR"}, workGroupConstraint);

    tuner.SetReferenceComputation(valuesOutId, [&valuesIn](void* buffer)
    {
        std::memcpy(buffer, valuesIn.data(), valuesIn.size() * sizeof(unsigned int));
        unsigned int* intArray = static_cast<unsigned int*>(buffer);
        std::sort(intArray, intArray + valuesIn.size());
    });

    const auto results = tuner.Tune(kernel);
    tuner.SaveResults(results, "Sort2Output", ktt::OutputFormat::JSON);

    return 0;
}
