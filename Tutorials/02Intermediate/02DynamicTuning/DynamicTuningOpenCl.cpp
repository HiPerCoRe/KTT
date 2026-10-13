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
        Beginning of code similar to KernelTuning
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

    // The input vectors are refilled before every kernel run, so KTT reads their data directly instead of keeping a copy
    // of their content at the time the arguments were created. The vectors must stay valid and must not reallocate while
    // the tuner uses them.
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalar);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

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

    // By default KTT makes a copy of the input data once and then doesn't update this copy. The intended solution is to update
    // input data explicitly in the kernel launcher. For scalars, it is the only solution. For vectors, another solution is to
    // set the argument to be referenced from user data, without making a KTT-managed copy (see overloads of AddArgumentVector),
    // and then disable caching of read-only arguments, so they are pulled from user data every (tuning) run.
    tuner.SetLauncher(kernel, [&a, &aId, &b, &bId, &scalar, &scalarId, &definition](ktt::ComputeInterface &interface){
        interface.UpdateBuffer(aId, a.data());
        interface.UpdateBuffer(bId, b.data());
        interface.UpdateScalarArgument(scalarId, &scalar);
        // Do not forget to also run the kernel definition.
        interface.RunKernel(definition);
    });

    while (SecondsSince(tuningStart) < tuningBudgetSeconds && iterationCount < maxIterations)
    {
        // Dynamic tuning uses the TuneIteration method. It runs a single configuration and returns a kernel result with timings, 
        // which could be used to e.g. determine if tuning should be stopped.
        // Second argument can contain descriptors of output buffers, much like RunKernel, so an application can already process
        // input while tuning. If input changes between iterations, it is necessary to set the flag for recomputing reference;
        // it may also be necessary to restart tuning altogether, since a different kind of input could have a different optimal
        // configuration. PreciseMeasurementParameters can also be set for more stable timings. Once all configurations have been 
        // launched, further calls launch the best configuration found so far.
        tuner.TuneIteration(kernel, {ktt::BufferOutputDescriptor(resultId, result.data())}, true);
        ++iterationCount;
        WriteResult(result);

        // Generating the reference output because of frequent input changes could be very slow. If one is confident that all
        // variants of a kernel are correct, output validation can be disabled with, for example:
        // tuner.SetValidationMode(ktt::ValidationMode::None);
        // It is also possible to not define a reference computation/kernel in the first place.

        // Note that this change only reaches the kernel because of explicit updates in the kernel launcher defined above.
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
        tuner.Run(kernel, bestConfiguration, {});
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
