// KTT tutorial demonstrating stop conditions.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include "Api/Searcher/RandomSearcher.h"
#include "Api/StopCondition/InterruptSignal.h"
#include "Api/StopCondition/StopCondition.h"
#include "Api/StopCondition/TuningDuration.h"
#include "Api/StopCondition/UnionCondition.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    /******************************************************
        Beginning of code identical to previous tutorial
    ******************************************************/
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/01Basic/03StopConditions/OpenClKernel.cl";

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
    /******************************************************
        End of identical code
    ******************************************************/

    // Modified from the previous tutorial to make tuning slower, so the effect of stop conditions is visible.
    tuner.AddParameter(kernel, "multiply_work_group_size", param_values);
    tuner.AddParameter(kernel, "REPETITIONS", std::vector<uint64_t>{2500, 5000, 10000});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size",
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // KTT stop conditions stop the tuning process when they are fulfilled. InterruptSignal stops when the program catches SIGINT,
    // TuningDuration stops after N seconds have passed, and UnionCondition stops when any one of its components is fulfilled.
    std::vector<std::shared_ptr<ktt::StopCondition>> unionedConditions = {
        std::make_shared<ktt::InterruptSignal>(),
        std::make_shared<ktt::TuningDuration>(30),
    };
    std::unique_ptr<ktt::StopCondition> condition = std::make_unique<ktt::UnionCondition>(unionedConditions);

    // The default deterministic searcher changes some variables faster than others, so in an incomplete search large "continuous"
    // parts of the tuning space might not be explored at all. RandomSearcher ensures uniform exploration.
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel, std::move(condition));

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
