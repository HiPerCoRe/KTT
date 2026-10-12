// KTT tutorial demonstrating stop conditions.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include "Api/Configuration/KernelConfiguration.h"
#include "Api/Searcher/Searcher.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

// A basic Searcher implementation.
class EvenFirst : public ktt::Searcher 
{
    uint64_t m_index;
    ktt::KernelConfiguration m_currentConfig;

    public:
    EvenFirst() : ktt::Searcher::Searcher() 
    {}

    void OnInitialize() override 
    {
        m_index = 0;
        // GetCurrentConfiguration should already return a valid configuration after OnInitialize
        m_currentConfig = GetConfiguration(m_index);
    }

    void OnReset() override 
    {
        m_index = 0;
    }

    // Other useful methods are: GetIndex (reverse of GetConfiguration), GetRandomConfiguration, GetNeighbourConfigurations,
    // GetUnexploredConfigurationsCount, and GetExploredIndices.
    bool CalculateNextConfiguration(const ktt::KernelResult &) override
    {
        m_index += 2;
        if (m_index >= GetConfigurationsCount()) 
        {
            if (m_index % 2 == 0) m_index = 1;
            else return false;
        }
        m_currentConfig = GetConfiguration(m_index);
        return true;
    }

    ktt::KernelConfiguration GetCurrentConfiguration() const override 
    {
        return m_currentConfig;
    }
};

int main(int argc, char** argv)
{
    /******************************************************
        Beginning of code identical to StopConditions
    ******************************************************/
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + KTT_TUTORIAL_KERNEL_FILE;

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
    // Work-group size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    const ktt::DimensionVector workGroupDimensions;
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalarValue);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    tuner.SetReferenceComputation(resultId, [&a, &b, &scalarValue](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < a.size(); ++i)
        {
            resultArray[i] = a[i] + b[i] + scalarValue;
        }
    });

    tuner.AddParameter(kernel, "multiply_work_group_size", std::vector<uint64_t>{32, 64, 128, 256});

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size",
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    /******************************************************
        End of identical code
    ******************************************************/
    
    // We may set the searcher to any implementation that inherits from the abstract Seacher class. An advanced user is relatively
    // easily able to create their own custom searcher or modify and existing one.
    tuner.SetSearcher(kernel, std::make_unique<EvenFirst>());

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
