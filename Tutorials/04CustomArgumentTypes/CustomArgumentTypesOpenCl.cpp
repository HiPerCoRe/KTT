#include <iostream>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

// Definition of custom data type, it needs to be defined in kernel source code as well.
struct KernelData
{
    float m_A;
    float m_B;
    float m_Result;
};

// Function which checks that two struct instances contain the same result. This is needed by the tuner to validate kernel arguments
// with custom data types.
bool CompareData(const void* resultPointer, const void* referencePointer)
{
    const auto* result = static_cast<const KernelData*>(resultPointer);
    const auto* reference = static_cast<const KernelData*>(referencePointer);

    if (result->m_Result != reference->m_Result)
    {
        std::cerr << "Result " << result->m_Result << " does not equal reference " << reference->m_Result << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/04CustomArgumentTypes/OpenClKernel.cl";

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

    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector ndRangeDimensions(numberOfElements);
    const ktt::DimensionVector workGroupDimensions;
    
    std::vector<KernelData> data(numberOfElements);

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        data[i].m_A = static_cast<float>(i);
        data[i].m_B = static_cast<float>(i + 1);
        data[i].m_Result = 0.0f;
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);

    const ktt::ArgumentId dataId = tuner.AddArgumentVector(data, ktt::ArgumentAccessType::ReadWrite);
    tuner.SetArguments(definition, {dataId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    tuner.SetReferenceComputation(dataId, [&data](void* buffer)
    {
        auto* resultArray = static_cast<KernelData*>(buffer);

        for (size_t i = 0; i < data.size(); ++i)
        {
            resultArray[i].m_Result = data[i].m_A + data[i].m_B;
        }
    });

    // Set the previously defined function as value comparator.
    tuner.SetValueComparator(dataId, CompareData);

    // Add parameter and thread modifier for the kernel. See the previous tutorial for more information.
    tuner.AddParameter(kernel, "multiply_work_group_size", std::vector<uint64_t>{32, 64, 128, 256});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size",
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
