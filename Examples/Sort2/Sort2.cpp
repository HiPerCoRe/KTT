#include "../ExampleReferenceComputation.h"
#include "KttTypes.h"
#include <assert.h>
#include <cstdint>

using namespace std;

class Sort2 : public ExampleReferenceComputation
{
protected:
    Sort2(
        std::shared_ptr<ExampleConfiguration> config,
        int defaultProblemSize,
        string exampleFolderPath, string defaultKernelFileBaseName
    ):
    ExampleReferenceComputation(config, defaultProblemSize,
        exampleFolderPath, defaultKernelFileBaseName
    )
    {
        // problem size is in MiB
        m_size = m_problemSize * 1024 * 1024 / sizeof(unsigned int);
    }

    friend ExampleBase;

    int m_size;
    const unsigned int SORT_BITS = 32;
    const unsigned int nbits = 4;

    // Input and output vectors
    vector<unsigned int> m_keysIn, m_keysOut;
    vector<unsigned int> m_valuesIn, m_valuesOut;

    ktt::KernelDefinitionId m_radixSortBlocks, m_findRadixOffsetsId, m_reorderDataId, m_vectorAddUniform4Id, m_scanId;

    // Needed as member variables to be usable in InitReference and lambdas
    ktt::ArgumentId m_valuesOutId;
    ktt::ArgumentId m_countersId;
    ktt::ArgumentId m_counterSumsId;
    ktt::ArgumentId m_blockOffsetsId;
    ktt::ArgumentId m_startBitId;
    ktt::ArgumentId m_scanNumBlocksId;
    ktt::ArgumentId m_numElementsId;
    ktt::ArgumentId m_fullBlockId;
    ktt::ArgumentId m_storeSumId;
    ktt::ArgumentId m_scanInDataId;
    ktt::ArgumentId m_scanOutDataId;
    ktt::ArgumentId m_scanOneBlockSumId;

    // Helper function for recursive scan
    void ScanArrayRecursive(ktt::ComputeInterface& interface,
        vector<unsigned int>& outArray, vector<unsigned int>& inArray, unsigned int numElements, int level,
        vector<vector<unsigned int>>& blockSums)
    {
        // Kernels handle 8 elems per thread
        const vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
        unsigned int scanBlockSize = (unsigned int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_BLOCK_SIZE");
        unsigned int sortVectorSize = (unsigned int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_VECTOR");
        const ktt::DimensionVector workGroupDimensions(scanBlockSize, 1, 1);
        unsigned int numBlocks = max(1u, (unsigned int)ceil((float)numElements/(sortVectorSize*scanBlockSize)));
        const ktt::DimensionVector ndRangeDimensions(numBlocks, 1, 1);

        interface.UpdateScalarArgument(m_numElementsId, &numElements);
        interface.UpdateBuffer(m_scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
        interface.UpdateBuffer(m_scanInDataId, inArray.data(), inArray.size() * sizeof(unsigned int));
        interface.UpdateBuffer(m_scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
        bool fullBlock = (numElements == numBlocks * sortVectorSize * scanBlockSize);
        interface.UpdateScalarArgument(m_fullBlockId, &fullBlock);
        bool storeSum;

        // execute the scan
        if (numBlocks > 1)
        {
            storeSum = 1;
            interface.UpdateScalarArgument(m_storeSumId, &storeSum);
            interface.RunKernel(m_scanId, ndRangeDimensions, workGroupDimensions);
            interface.DownloadBuffer(m_scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
            interface.DownloadBuffer(m_scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));

            ScanArrayRecursive(interface, blockSums[level], blockSums[level], numBlocks, level + 1, blockSums);

            interface.UpdateScalarArgument(m_numElementsId, &numElements);
            interface.UpdateBuffer(m_scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
            interface.UpdateBuffer(m_scanOneBlockSumId, blockSums.at(level).data(), blockSums.at(level).size() * sizeof(unsigned int));
            interface.RunKernel(m_vectorAddUniform4Id, ndRangeDimensions, workGroupDimensions);
            interface.DownloadBuffer(m_scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
        }
        else
        {
            storeSum = 0;
            interface.UpdateScalarArgument(m_storeSumId, &storeSum);
            interface.RunKernel(m_scanId, ndRangeDimensions, workGroupDimensions);
            interface.DownloadBuffer(m_scanOutDataId, outArray.data(), outArray.size() * sizeof(unsigned int));
        }
    }

    void InitData() override
    {
        // Create input and output vectors and initialize with pseudorandom numbers
        m_keysIn.resize(m_size);
        m_keysOut.resize(m_size);
        m_valuesIn.resize(m_size);
        m_valuesOut.resize(m_size);

        FillBuffers<unsigned int>({&m_keysIn}, 0, 1023);
        m_valuesIn = m_keysIn;
    }

    void InitKernel() override
    {
        // Declare kernels and their dimensions
        const ktt::DimensionVector ndRangeDimensions;
        const ktt::DimensionVector workGroupDimensions;

        m_radixSortBlocks = m_tuner->AddKernelDefinitionFromFile("radixSortBlocks", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_findRadixOffsetsId = m_tuner->AddKernelDefinitionFromFile("findRadixOffsets", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_reorderDataId = m_tuner->AddKernelDefinitionFromFile("reorderData", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_vectorAddUniform4Id = m_tuner->AddKernelDefinitionFromFile("vectorAddUniform4", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_scanId = m_tuner->AddKernelDefinitionFromFile("scan", m_kernelFile, ndRangeDimensions, workGroupDimensions);

        // Add arguments for kernels
        // All parameters with foo values (empty vectors or scalar 1) will be updated in tuning manipulator, as their value depends on tuning parameters
        const ktt::ArgumentId nbitsId = m_tuner->AddArgumentScalar(nbits);
        m_startBitId = m_tuner->AddArgumentScalar(0);
        const ktt::ArgumentId sizeId = m_tuner->AddArgumentScalar(m_size);

        const ktt::ArgumentId keysOutId = m_tuner->AddArgumentVector(m_keysOut, ktt::ArgumentAccessType::ReadWrite);
        m_valuesOutId = m_tuner->AddArgumentVector(m_valuesOut, ktt::ArgumentAccessType::ReadWrite);
        const ktt::ArgumentId keysInId = m_tuner->AddArgumentVector(m_keysIn, ktt::ArgumentAccessType::ReadWrite);
        const ktt::ArgumentId valuesInId = m_tuner->AddArgumentVector(m_valuesIn, ktt::ArgumentAccessType::ReadWrite);

        m_countersId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
        m_counterSumsId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
        m_blockOffsetsId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);

        m_scanNumBlocksId = m_tuner->AddArgumentScalar(1);
        m_numElementsId = m_tuner->AddArgumentScalar(1);

        m_scanOutDataId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
        m_scanInDataId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadOnly);
        m_scanOneBlockSumId = m_tuner->AddArgumentVector(vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
        m_fullBlockId = m_tuner->AddArgumentScalar(1);
        m_storeSumId = m_tuner->AddArgumentScalar(1);

        m_kernel = m_tuner->CreateCompositeKernel("Sort", 
            {m_radixSortBlocks, m_findRadixOffsetsId, m_reorderDataId, m_vectorAddUniform4Id, m_scanId}, 
            [this, nbitsId, keysOutId, keysInId, valuesInId,
            sizeId](ktt::ComputeInterface& interface)
        {
            const vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();

            int sortBlockSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_BLOCK_SIZE");
            int sortVectorSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SORT_VECTOR");
            const ktt::DimensionVector workGroupDimensionsSort(sortBlockSize, 1, 1);
            const ktt::DimensionVector ndRangeDimensionsSort(m_size / sortVectorSize / sortBlockSize, 1, 1);

            int scanBlockSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_BLOCK_SIZE");
            int scanVectorSize = (int)ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SCAN_VECTOR");
            const ktt::DimensionVector workGroupDimensionsScan(scanBlockSize, 1, 1);
            const ktt::DimensionVector ndRangeDimensionsScan(m_size / scanVectorSize / scanBlockSize, 1, 1);

            unsigned int scanNumBlocks = static_cast<unsigned int>(ndRangeDimensionsScan.GetSizeX());
            interface.UpdateScalarArgument(m_scanNumBlocksId, &scanNumBlocks);

            unsigned int countersSizeVal = 16 * scanNumBlocks;
            vector<unsigned int> counters(countersSizeVal);
            interface.ResizeBuffer(m_countersId, countersSizeVal * sizeof(unsigned int), false);
            interface.ResizeBuffer(m_scanInDataId, countersSizeVal * sizeof(unsigned int), false);
            interface.UpdateBuffer(m_countersId, counters.data());

            vector<unsigned int> counterSums(countersSizeVal);
            interface.ResizeBuffer(m_counterSumsId, countersSizeVal * sizeof(unsigned int), false);
            interface.ResizeBuffer(m_scanOutDataId, countersSizeVal * sizeof(unsigned int), false);
            interface.UpdateBuffer(m_counterSumsId, counterSums.data());

            vector<unsigned int> blockOffsets(countersSizeVal);
            interface.ResizeBuffer(m_blockOffsetsId, countersSizeVal * sizeof(unsigned int), false);
            interface.UpdateBuffer(m_blockOffsetsId, blockOffsets.data());

            // Allocate space for block sums in the scan kernel.
            unsigned int maxNumScanElements = m_size;
            unsigned int numScanElts = maxNumScanElements;

            vector<vector<unsigned int>> scanBlockSums;

            do
            {
                unsigned int numBlocks = max(1, (int)ceil((float)numScanElts / (sortVectorSize * scanBlockSize)));

                if (numBlocks > 1)
                {
                    scanBlockSums.push_back(vector<unsigned int>(numBlocks));
                }

                numScanElts = numBlocks;
            }
            while (numScanElts > 1);

            scanBlockSums.push_back(vector<unsigned int>(1));
            interface.ResizeBuffer(m_scanOneBlockSumId, scanBlockSums[0].size() * sizeof(unsigned int), false);

            unsigned int startbit;
            bool swap = true;

            for (startbit = 0; startbit < SORT_BITS; startbit += nbits)
            {
                assert(startbit < 32);
                assert(startbit + nbits <= 32);

                interface.UpdateScalarArgument(m_startBitId, &startbit);

                //radixSortBlocks
                //  <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
                //  (nbits, startbit, tempKeys, tempValues, keys, values);
                interface.RunKernel(m_radixSortBlocks, ndRangeDimensionsSort, workGroupDimensionsSort);

                //findRadixOffsets
                //  <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
                //  ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
                //   findBlocks);
                interface.RunKernel(m_findRadixOffsetsId, ndRangeDimensionsScan, workGroupDimensionsScan);
                interface.DownloadBuffer(m_countersId, counters.data());

                ScanArrayRecursive(interface, counterSums, counters, 16 * scanNumBlocks, 0, scanBlockSums);

                //reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
                //  (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
                //   (uint2*)tempValues, blockOffsets, countersSum, counters,
                //   reorderBlocks);
                interface.UpdateBuffer(m_counterSumsId, counterSums.data(), counterSums.size() * sizeof(unsigned int));
                interface.UpdateBuffer(m_countersId, counters.data(), counters.size() * sizeof(unsigned int));

                if (swap)
                {
                    interface.SwapArguments(m_reorderDataId, keysOutId, keysInId);
                    interface.SwapArguments(m_reorderDataId, m_valuesOutId, valuesInId);
                    swap = !swap;
                }

                interface.RunKernel(m_reorderDataId, ndRangeDimensionsScan, workGroupDimensionsScan);
            }
        });

        //radixSortBlocks
        m_tuner->SetArguments(m_radixSortBlocks, {nbitsId, m_startBitId, keysOutId, m_valuesOutId, keysInId, valuesInId});

        //findRadixOffsets
        m_tuner->SetArguments(m_findRadixOffsetsId, {keysOutId, m_countersId, m_blockOffsetsId, m_startBitId, sizeId, m_scanNumBlocksId});

        //reorderData
        m_tuner->SetArguments(m_reorderDataId, {m_startBitId, keysOutId, m_valuesOutId, keysInId, valuesInId, m_blockOffsetsId, m_counterSumsId, m_countersId, m_scanNumBlocksId});

        //vectorAddUniform
        m_tuner->SetArguments(m_vectorAddUniform4Id, {m_scanOutDataId, m_scanOneBlockSumId, m_numElementsId});

        //scan
        m_tuner->SetArguments(m_scanId, {m_scanOutDataId, m_scanInDataId, m_scanOneBlockSumId, m_numElementsId, m_fullBlockId, m_storeSumId});
    }

    void InitTuningSpace() override
    {
        m_tuner->AddParameter(m_kernel, "SORT_BLOCK_SIZE", vector<uint64_t>{32, 64, 128, 256, 512, 1024});
        m_tuner->AddParameter(m_kernel, "SCAN_BLOCK_SIZE", vector<uint64_t>{32, 64, 128, 256, 512, 1024});
        m_tuner->AddParameter(m_kernel, "SORT_VECTOR", vector<uint64_t>{2, 4, 8});
        m_tuner->AddParameter(m_kernel, "SCAN_VECTOR", vector<uint64_t>{2, 4, 8});

        auto workGroupConstraint = [](const vector<uint64_t>& vector) {return (float)vector.at(1) / vector.at(0) == (float)vector.at(2) / vector.at(3) &&
            !(vector.at(0) == 1024 && vector.at(1) == 1024 && vector.at(2) == 8 && vector.at(3) == 8);};
        m_tuner->AddConstraint(m_kernel, {"SORT_BLOCK_SIZE", "SCAN_BLOCK_SIZE", "SORT_VECTOR", "SCAN_VECTOR"}, workGroupConstraint);
    }

    void InitReference() override
    {
        m_tuner->SetReferenceComputation(m_valuesOutId, [this](void* buffer)
        {
            memcpy(buffer, m_valuesIn.data(), m_valuesIn.size() * sizeof(unsigned int));
            unsigned int* intArray = static_cast<unsigned int*>(buffer);
            sort(intArray, intArray + m_valuesIn.size());
        });
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Sort2> sort2 = Sort2::Create<Sort2>(argc, argv, 32, "Examples/Sort2", "Sort2");
    sort2->Run();

    return 0;
}
