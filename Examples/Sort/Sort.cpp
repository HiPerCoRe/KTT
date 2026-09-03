#include "../ExampleReferenceComputation.h"

using namespace std;

class Sort : public ExampleReferenceComputation
{
protected:
    Sort(int argc, char **argv, int defaultProblemSize, string exampleFolderPath,
         string defaultKernelFileBaseName) :
        ExampleReferenceComputation(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        m_size = m_problemSize * 1024 * 1024 / sizeof(unsigned int);
    }

    friend ExampleBase;

    uint32_t m_size;
    vector<unsigned int> m_in, m_out;

    ktt::ArgumentId m_inId;
    ktt::ArgumentId m_outId;
    ktt::ArgumentId m_sizeId;
    ktt::ArgumentId m_shiftId;
    ktt::ArgumentId m_isumsId;
    ktt::ArgumentId m_numberOfGroupsId;

    ktt::KernelDefinitionId m_reduceDefinition;
    ktt::KernelDefinitionId m_topScanDefinition;
    ktt::KernelDefinitionId m_bottomScanDefinition;

    void InitData() override
    {
        // Create input and output vectors and initialize with pseudorandom numbers
        m_in.resize(m_size);
        m_out.resize(m_size);

        FillBuffers<unsigned int>({&m_in});
    }

    void InitKernel() override
    {
        // Declare kernels and their dimensions
        const ktt::DimensionVector ndRangeDimensions;
        const ktt::DimensionVector workGroupDimensions;

        m_reduceDefinition = m_tuner->AddKernelDefinitionFromFile("reduce", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_topScanDefinition = m_tuner->AddKernelDefinitionFromFile("top_scan", m_kernelFile, workGroupDimensions, workGroupDimensions);
        m_bottomScanDefinition = m_tuner->AddKernelDefinitionFromFile("bottom_scan", m_kernelFile, ndRangeDimensions, workGroupDimensions);

        // Add arguments for kernels
        m_inId = m_tuner->AddArgumentVector(m_in, ktt::ArgumentAccessType::ReadWrite);
        m_outId = m_tuner->AddArgumentVector(m_out, ktt::ArgumentAccessType::ReadWrite);
        m_sizeId = m_tuner->AddArgumentScalar(m_size);
        int shift = 0;
        m_shiftId = m_tuner->AddArgumentScalar(shift); // Will be updated as the kernel execution is iterative

        int numberOfGroups = 1;
        int isumsSize = 16 * numberOfGroups;
        // Vector argument will be updated in tuning manipulator as its size depends on the number of work-groups
        m_isumsId = m_tuner->AddArgumentVector(vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite);
        m_numberOfGroupsId = m_tuner->AddArgumentScalar(numberOfGroups);

        m_kernel = m_tuner->CreateCompositeKernel("Sort", {m_reduceDefinition, m_topScanDefinition, m_bottomScanDefinition},
            [this](ktt::ComputeInterface& interface)
        {
            const int radix_width = 4;
            const vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
            uint64_t localSize = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "LOCAL_SIZE");
            uint64_t globalSize = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "GLOBAL_SIZE");

            int numberOfGroups = static_cast<int>(globalSize / localSize);
            interface.UpdateScalarArgument(m_numberOfGroupsId, &numberOfGroups);
            int isumsSize = 16 * numberOfGroups;

            // Vector, read-write, must be added after global and local size are determined, as its size depends on the number of groups
            interface.ResizeBuffer(m_isumsId, isumsSize * sizeof(unsigned int), false);

            bool inOutSwapped = false;

            for (int shift = 0; shift < static_cast<int>(sizeof(unsigned int) * 8); shift += radix_width)
            {
                // Like scan, we use a reduce-then-scan approach

                // But before proceeding, update the shift appropriately for each kernel. This is how many bits to shift to the right used in binning.
                interface.UpdateScalarArgument(m_shiftId, &shift);

                // Also, the sort is not in place, so swap the input and output buffers on each pass.
                const bool even = ((shift / radix_width) % 2 == 0) ? true : false;

                if (even)
                {
                    interface.ChangeArguments(m_reduceDefinition, {m_inId, m_isumsId, m_sizeId, m_shiftId});
                }
                else
                {
                    interface.ChangeArguments(m_reduceDefinition, {m_outId, m_isumsId, m_sizeId, m_shiftId});
                }

                // Each thread block gets an equal portion of the input array, and computes occurrences of each digit.
                interface.RunKernel(m_reduceDefinition);

                // Next, a top-level exclusive scan is performed on the per block histograms. This is done by a single work group
                // (note global size here is the same as local).
                interface.RunKernel(m_topScanDefinition);

                // Finally, a bottom-level scan is performed by each block that is seeded with the scanned histograms which rebins,
                // locally scans, then scatters keys to global memory
                interface.RunKernel(m_bottomScanDefinition);

                // Also, the sort is not in place, so swap the input and output buffers on each pass.
                interface.SwapArguments(m_bottomScanDefinition, m_inId, m_outId);

                if (shift + radix_width < static_cast<int>(sizeof(unsigned int) * 8)) // Not the last iteration
                {
                    inOutSwapped = !inOutSwapped;
                }
            }

            if (inOutSwapped)
            {
                // Copy contents of in to out, since they are swapped
                interface.CopyBuffer(m_outId, m_inId);
            }
        });

        m_tuner->SetArguments(m_reduceDefinition, {m_inId, m_isumsId, m_sizeId, m_shiftId});
        m_tuner->SetArguments(m_topScanDefinition, {m_isumsId, m_numberOfGroupsId});
        m_tuner->SetArguments(m_bottomScanDefinition, {m_inId, m_isumsId, m_outId, m_sizeId, m_shiftId});
    }

    void InitTuningSpace() override
    {
        // Parameter for the length of OpenCL vector data types used in the kernels
        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner->AddParameter(m_kernel, "FPVECTNUM", vector<uint64_t>{4, 8, 16});
        }
        else
        {
            m_tuner->AddParameter(m_kernel, "FPVECTNUM", vector<uint64_t>{4});
        }

        // Local size below 128 does not work correctly, not even with the benchmark code
        m_tuner->AddParameter(m_kernel, "LOCAL_SIZE", vector<uint64_t>{128, 256, 512});
        m_tuner->AddThreadModifier(m_kernel, {m_reduceDefinition, m_topScanDefinition, m_bottomScanDefinition},
            ktt::ModifierType::Local, ktt::ModifierDimension::X, "LOCAL_SIZE", ktt::ModifierAction::Multiply);

        // Second kernel global size is always equal to local size
        m_tuner->AddParameter(m_kernel, "GLOBAL_SIZE", vector<uint64_t>{512, 1024, 2048, 4096, 8192, 16384, 32768});
        m_tuner->AddThreadModifier(m_kernel, {m_reduceDefinition, m_bottomScanDefinition},
            ktt::ModifierType::Global, ktt::ModifierDimension::X, "GLOBAL_SIZE", ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_reduceDefinition, m_bottomScanDefinition},
            ktt::ModifierType::Global, ktt::ModifierDimension::X, "LOCAL_SIZE",
            ktt::ModifierAction::Divide);

        auto workGroupConstraint = [](const vector<uint64_t>& vector) {return vector.at(0) != 128 || vector.at(1) != 32768;};
        m_tuner->AddConstraint(m_kernel, {"LOCAL_SIZE", "GLOBAL_SIZE"}, workGroupConstraint);
    }

    void InitReference() override
    {
        m_tuner->SetReferenceComputation(m_outId, [this](void* buffer)
        {
            memcpy(buffer, m_in.data(), m_in.size() * sizeof(unsigned int));
            unsigned int* intArray = static_cast<unsigned int*>(buffer);
            sort(intArray, intArray + m_in.size());
        });
    }
};

int main(int argc, char** argv)
{
    unique_ptr<Sort> sort = Sort::Create<Sort>(argc, argv, 32, "Examples/Sort", "Sort");
    sort->Run();

    return 0;
}
