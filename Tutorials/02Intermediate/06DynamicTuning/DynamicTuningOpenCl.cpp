// KTT tutorial demonstrating dynamic tuning, in which the program tunes the kernel and uses its output at the same
// time instead of tuning everything before the actual work starts.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include "Api/ComputeInterface.h"
#include "Api/Output/BufferOutputDescriptor.h"
#include "Api/Searcher/RandomSearcher.h"
#include "KernelRunner/ValidationMode.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

// Returns the number of seconds that have passed since the given time point.
double SecondsSince(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Simulates real input.
void GetNewData(size_t inputSize, std::vector<float> &a, std::vector<float> &b, float &scalar)
{
    a.resize(inputSize);
    b.resize(inputSize);
    scalar = static_cast<float>(rand() % 32);

    for (size_t i = 0; i < inputSize; ++i)
    {
        a[i] = static_cast<float>(rand() % 32);
        b[i] = static_cast<float>(rand() % 32);
    }
}

// Placeholder that demonstrates how result data could be used
void WriteResult(const std::vector<float> &result)
{
    std::ofstream outFile("out.bin", std::ios::binary | std::ios::out);
    outFile.write((char *) result.data(), result.size() * sizeof(float));
}

int main(int argc, char** argv)
{
    /******************************************************
        Beginning of code similar to KernelRunning
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
    const ktt::DimensionVector workGroupDimensions;
    
    std::vector<float> a;
    std::vector<float> b;
    float scalar;
    GetNewData(numberOfElements, a, b, scalar);
    std::vector<float> result(numberOfElements, 0.0f);

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);

    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);

    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalar);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // There is no need to update input buffers as that is done before every tuning iteration in any case.
    // The first time a buffer is updated, KTT automatically allocates memory on the device before writing data. 
    // On subsequent updates, it only writes the data.
    tuner.UpdateBuffer(resultId);

    tuner.SetReferenceComputation(resultId, [&a, &b, &scalar](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < a.size(); ++i)
        {
            resultArray[i] = a[i] + b[i] + scalar;
        }
    });

    /******************************************************
        End of similar code
    ******************************************************/

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Tuning space is artificially inflated to make the exploration noticeably long, similar to StopConditions
    tuner.AddParameter(kernel, "multiply_work_group_size", std::vector<uint64_t>{2, 4, 8, 16, 32, 64, 128, 256, 512});
    tuner.AddParameter(kernel, "REPETITIONS", std::vector<uint64_t>{250, 500, 1000});
    tuner.AddParameter(kernel, "DUMMY_A", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    tuner.AddParameter(kernel, "DUMMY_B", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
        "multiply_work_group_size", ktt::ModifierAction::Multiply);

    // Every kernel run of the tuning loop would otherwise log an information message, which would bury the printed
    // statistics.
    ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Warning);

    // Choosing when to stop tuning and start exploiting the best known configuration is a complex problem outside the
    // scope of this tutorial, so here is only a very simple solution.
    const double tuningBudgetSeconds = 10.0;
    const double runningBudgetSeconds = 10.0;

    // The number of configurations of a kernel is only known once its configuration data exists, which the first
    // TuneIteration call would create as well.
    tuner.InitializeConfigurationData(kernel);
    const uint64_t maxIterations = tuner.GetConfigurationsCount(kernel);
    std::cout << "--- Tuning " << maxIterations << " configurations for "
              << tuningBudgetSeconds << " seconds ---" << std::endl;

    const auto tuningStart = std::chrono::steady_clock::now();
    auto lastProgressPrint = tuningStart;
    uint64_t iterationCount = 0;

    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());

    // Regenerating the reference output because of frequent input changes and comparing against it could be very slow. 
    // If one is confident that all variants of a kernel are correct, output validation can be disabled with, 
    // for example:
    tuner.SetValidationMode(ktt::ValidationMode::None);
    // It is also possible to not define a reference computation/kernel in the first place.
    // Disabling validation for dynamic tuning (at least in release configurations) is intended, to increase
    // performance.


    while (SecondsSince(tuningStart) < tuningBudgetSeconds && iterationCount < maxIterations)
    {
        // As per KernelRunning, it is recommended to use Tuner argument management methods to manually handle data.
        //
        // In reality, when dynamic tuning is integrated into an application, there will already be a buffer management
        // system in place. See ComputeApiInitializer in the Advanced tutorials for this.
        tuner.UpdateBuffer(aId);
        tuner.UpdateBuffer(bId);

        // Dynamic tuning uses the TuneIteration method. It runs a single configuration and returns a kernel result with timings, 
        // which could be used to e.g. determine if tuning should be stopped.
        // Second argument can contain descriptors of output buffers, much like RunKernel, so an application can already process
        // input while tuning. If input changes between iterations, it is necessary to set the flag for recomputing reference (in the case that validation is enabled);
        // it may also be necessary to restart tuning altogether, since a different kind of input could have a different optimal
        // configuration. PreciseMeasurementParameters can also be set for more stable timings. Once all configurations have been 
        // launched, further calls launch the best configuration found so far.
        tuner.TuneIteration(kernel, {ktt::BufferOutputDescriptor(resultId, result.data())}, true);
        ++iterationCount;
        WriteResult(result);

        // Note that this change only reaches the kernel because of explicit updates with Tuner::UpdateBuffer(...).
        GetNewData(numberOfElements, a, b, scalar);

        if (SecondsSince(lastProgressPrint) >= 1)
        {
            std::cout << "Explored " << iterationCount << "/" << maxIterations << " configurations.\n";
            lastProgressPrint = std::chrono::steady_clock::now();
        }
    }
    std::cout << "\n--- Tuning complete ---" << std::endl;
    std::cout << "Total runs: " << iterationCount << std::endl;
    std::cout << "Throughput: " << iterationCount / SecondsSince(tuningStart) << " runs/s" << std::endl;

    const ktt::KernelConfiguration bestConfiguration = tuner.GetBestConfiguration(kernel);

    std::cout << "Best known configuration is " << bestConfiguration.GetString() << "\n";
    std::cout << "\n--- Running the best configuration for " << runningBudgetSeconds << " seconds ---" << std::endl;

    const auto runningStart = std::chrono::steady_clock::now();
    lastProgressPrint = runningStart;
    iterationCount = 0;

    while (SecondsSince(runningStart) < runningBudgetSeconds)
    {
        // Compiled kernel definitions are saved in a cache, so repeated runs of the same configuration avoid
        // having to recompile it. As a result, the overhead of using KTT for running kernels is negligible.
        tuner.UpdateBuffer(aId);
        tuner.UpdateBuffer(bId);
        tuner.Run(kernel, bestConfiguration, {ktt::BufferOutputDescriptor(resultId, result.data())});
        WriteResult(result);
        ++iterationCount;

        if (SecondsSince(lastProgressPrint) >= 1)
        {
            std::cout << "Run " << iterationCount << " | Elapsed: " << SecondsSince(runningStart) << "s of "
                << runningBudgetSeconds << "s" << std::endl;
            lastProgressPrint = std::chrono::steady_clock::now();
        }
    }

    std::cout << "\n--- Running complete ---" << std::endl;
    std::cout << "Total runs: " << iterationCount << std::endl;
    std::cout << "Throughput: " << iterationCount / SecondsSince(runningStart) << " runs/s" << std::endl;

    tuner.Run(kernel, bestConfiguration, {ktt::BufferOutputDescriptor(resultId, result.data())});

    std::cout << "\nPrinting the first 10 elements from result: ";

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }

    std::cout << std::endl;
    return 0;
}
