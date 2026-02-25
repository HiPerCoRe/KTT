#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

const std::string defaultKernelFile = kernelPrefix + "../Examples/CppSimple/CppSimpleKernel.cppkernel";
const auto computeApi = ktt::ComputeApi::Cpp;

int main()
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;

    // Create tuner
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Define kernel dimensions (global size, work group size)
    const size_t elementCount = 1024 * 1024; // 1M elements for meaningful timing
    const ktt::DimensionVector ndRangeDimensions(elementCount);
    const ktt::DimensionVector workGroupDimensions(1);

    // Kernel definition from file
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile(
        "vectorAdd",
        kernelFile,
        ndRangeDimensions,
        workGroupDimensions
    );

    // Create simple kernel
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("VectorAdd", definition);

    // Create data
    std::vector<float> a(elementCount);
    std::vector<float> b(elementCount);
    std::vector<float> out(elementCount, 0.0f);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < elementCount; ++i)
    {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    // Add arguments
    const ktt::ArgumentId argOut = tuner.AddArgumentVector(out, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId argA = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId argB = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);

    // Associate arguments with kernel definition
    tuner.SetArguments(definition, {argOut, argA, argB});

    // Set reference computation for validation
    tuner.SetReferenceComputation(argOut, [&a, &b](void* buffer)
    {
        float* result = static_cast<float*>(buffer);
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] + b[i];
        }
    });

    // Set validation method
    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

    // Add tuning parameter: UNROLL_FACTOR with values 1, 2, 4, 8, 16
    // This creates 5 different configurations to test
    tuner.AddParameter(kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8, 16});

    // Set launcher (just run kernel)
    tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
    {
        interface.RunKernel(definition);
    });

    // Tune with all configurations
    const auto results = tuner.Tune(kernel);

    // Print results
    std::cout << "\n=== Tuning Results ===" << std::endl;
    std::cout << "Total configurations tested: " << results.size() << std::endl;
    
    bool allPassed = true;
    for (const auto& result : results)
    {
        std::cout << "Configuration: " << result.GetConfiguration().GetString()
                  << " -> Duration: " << result.GetTotalDuration() << " us"
                  << " -> Status: " << (result.GetStatus() == ktt::ResultStatus::Ok ? "OK" : "FAILED")
                  << std::endl;
        if (result.GetStatus() != ktt::ResultStatus::Ok)
        {
            allPassed = false;
        }
    }

    if (allPassed && !results.empty())
    {
        std::cout << "\nC++ backend tuning test passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nC++ backend tuning test failed!" << std::endl;
        return 1;
    }
}
