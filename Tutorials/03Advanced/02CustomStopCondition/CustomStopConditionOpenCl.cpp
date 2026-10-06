// KTT tutorial demonstrating stop conditions.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

using std::chrono::system_clock;
using std::chrono::duration;

class ConsolePrompt : public ktt::StopCondition 
{
    const std::chrono::seconds m_period;
    system_clock::time_point m_lastPrompt;
    bool m_shouldPrompt = false;
public:
    ConsolePrompt(int periodSeconds) :
        m_period(periodSeconds),
        m_lastPrompt(system_clock::now())
    {}

    bool IsFulfilled() const
    {
        if (!m_shouldPrompt) return false;
        std::cout << "Continue tuning? ([y]/n)" << std::endl;
        std::string answer;
        std::getline(std::cin, answer);
        return answer == "n";
    }

    void Initialize(const uint64_t)
    {
    }

    void Update([[maybe_unused]] const ktt::KernelResult& result)
    {
        auto now = system_clock::now();
        if (m_shouldPrompt) m_lastPrompt = now;
        m_shouldPrompt = (now - m_lastPrompt > m_period);
    }
        
    std::string GetStatusString() const
    {
        return "Last prompt " + std::to_string(duration<float>(system_clock::now() - m_lastPrompt).count()) + " seconds ago.";
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

    std::vector<uint64_t> param_values(9);
    uint64_t exp = 2;
    for (uint64_t i = 0; i < param_values.size(); ++i) 
    {
        param_values[i] = exp;
        exp *= 2;
    }

    // Modified from the previous tutorial to make tuning slower, so the effect of stop conditions is visible.
    tuner.AddParameter(kernel, "multiply_work_group_size", param_values);
    tuner.AddParameter(kernel, "REPETITIONS", std::vector<uint64_t>{2500, 5000, 10000});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size",
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    /******************************************************
        End of identical code
    ******************************************************/

    std::unique_ptr<ktt::StopCondition> condition = std::make_unique<ConsolePrompt>(5);

    // The default deterministic searcher changes some variables faster than others, so in an incomplete search large "continuous"
    // parts of the tuning space might not be explored at all. RandomSearcher ensures uniform exploration.
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel, std::move(condition));

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
