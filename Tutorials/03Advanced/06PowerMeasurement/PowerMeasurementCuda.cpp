// KTT tutorial demonstrating kernel setup and tuning via KTT
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include "Api/Configuration/PreciseMeasurementParameters.h"
#include "Api/Output/KernelResult.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <ostream>
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
        Beginning of code identical to PreciseMeasurement (converted to CUDA)
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
    const ktt::DimensionVector gridDimensions(numberOfElements);
    const ktt::DimensionVector blockDimensions;
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::CUDA);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions,
        blockDimensions);
    
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

    tuner.AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256});

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Divide);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    ktt::PreciseMeasurementParameters preciseParams(2000, 20000, 0.005, ktt::DurationCalculationMethod::Median);

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel, nullptr, preciseParams);
    /******************************************************
        End of identical code
    ******************************************************/

    // With --power-usage enabled, the above-defined preciseParams mean that, when measuring a configuration, the tuner
    // repeats kernel launches until power consumption stabilizes or until maxTimeMs runs out.
    // In detail, it considers K most recent measurements, where K is max(2, number of kernel runs
    // completed before minTimeMs ran out), and calculates the maximum relative difference from the window's average
    // power consumption: max( abs(x_i - x_avg) / x_avg ) over all x_i in the window. 
    // If this value is not over maxPowerDiff, the measurement is considered to be stable and is terminated. 
    // If it does not stabilize before maxTimeMs, it is terminated anyway.

    // NOTE: This means that minTimeMs must be long enough that a sufficient number of kernel runs can fit into the 
    // stability measurement window.

    // NOTE: The minimum/median/average measurement is taken from ALL measurements, not just the considered window.
    // With power consumption, taking the minimum does not make sense in this case, as that will almost always be right
    // at the start of measuring, when the GPU is still cold.

    // Currently, one must obtain the most energy-efficient configuration manually -- GetBestConfiguration only uses
    // total duration. The default searchers also only use total duration (except deterministic and random, which do 
    // not use any metric). This should be addressed in an upcoming update.
    size_t bestResultIndex = -1;
    double bestEnergyConsumption = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < results.size(); ++i)
    {
        // KernelResult itself does not have power usage data, one must compute them from the ComputationResult vector.
        // This vector has information from every kernel definition launch made by the kernel launcher separately.
        // This kernel only launches a single definition once, so taking the first element is valid.
        auto computationResult = results[i].GetResults()[0];
        assert(computationResult.HasPowerData() && "Power measurement data was not collected. Pass --power-usage to "
            "premake when generating makefiles for KTT");
        double energyConsumption = computationResult.GetEnergyConsumption();
        if (energyConsumption < bestEnergyConsumption) 
        {
            bestEnergyConsumption = energyConsumption;
            bestResultIndex = i;
        }
    }
    std::cout << "Lowest energy consumption: " << bestEnergyConsumption << " with configuration " 
        << results[bestResultIndex].GetConfiguration().GetString() << std::endl;

    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
