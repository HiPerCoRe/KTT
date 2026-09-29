// KTT tutorial demonstrating kernel setup and tuning via KTT
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
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
        Beginning of code identical to KernelTuning
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

    // Measuring the tuning time makes it possible to report how many configurations per second were explored.
    const auto tuningStart = std::chrono::steady_clock::now();

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    // Count the runs that finished successfully and remember the fastest one of them. Runs that failed
    // are saved as well, but not used for finding the best configuration.
    const int totalRuns = static_cast<int>(results.size());
    int successfulRuns = 0;
    double bestDuration = std::numeric_limits<double>::max();
    std::string bestConfig;

    for (const ktt::KernelResult& result : results)
    {
        if (result.GetStatus() == ktt::ResultStatus::Ok)
        {
            successfulRuns++;

            if (result.GetTotalDuration() < bestDuration)
            {
                bestDuration = result.GetTotalDuration();
                bestConfig = result.GetConfiguration().GetString();
            }
        }
    }

    const auto tuningEnd = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(tuningEnd - tuningStart).count();
    const double throughput = elapsedSeconds > 0 ? static_cast<double>(totalRuns) / elapsedSeconds : 0;

    std::cout << "\n--- Offline tuning complete ---\n";
    std::cout << "Total runs: " << totalRuns << "\n";
    std::cout << "Successful runs: " << successfulRuns << "/" << totalRuns << "\n";
    if (!bestConfig.empty())
    {
        // GetTotalDuration always returns nanoseconds, also when the saved results use another time unit.
        std::cout << "Best configuration: " << bestConfig << "\n";
        std::cout << "Best duration: " << bestDuration << " ns\n";
    }
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " runs/s" << "\n";

    return 0;
}
