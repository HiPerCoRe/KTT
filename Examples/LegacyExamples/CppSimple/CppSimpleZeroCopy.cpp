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
    ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Debug);

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

    // Create data using malloc (user-managed memory for zero-copy)
    const size_t bufferSize = elementCount * sizeof(float);
    float* a = (float*)malloc(bufferSize);
    float* b = (float*)malloc(bufferSize);
    float* out = (float*)malloc(bufferSize);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < elementCount; ++i)
    {
        a[i] = dist(gen);
        b[i] = dist(gen);
        out[i] = 0.0f;
    }

    std::cout << "Testing zero-copy mode with user-managed buffers..." << std::endl;

    // Add arguments using zero-copy API (user-managed buffers)
    // Note: For C++ backend, ComputeBuffer is just void*
    std::cout << "User buffer addresses:" << std::endl;
    std::cout << "  out = " << out << std::endl;
    std::cout << "  a = " << a << std::endl;
    std::cout << "  b = " << b << std::endl;
    
    const ktt::ArgumentId argOut = tuner.AddArgumentVector<float>(
        out, bufferSize, ktt::ArgumentAccessType::WriteOnly,
        ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId argA = tuner.AddArgumentVector<float>(
        a, bufferSize, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId argB = tuner.AddArgumentVector<float>(
        b, bufferSize, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Host);
    
    std::cout << "Argument IDs:" << std::endl;
    std::cout << "  argOut = " << argOut << std::endl;
    std::cout << "  argA = " << argA << std::endl;
    std::cout << "  argB = " << argB << std::endl;

    // Associate arguments with kernel definition
    tuner.SetArguments(definition, {argOut, argA, argB});

    // Set reference computation for validation
    tuner.SetReferenceComputation(argOut, [a, b, elementCount](void* buffer)
    {
        float* result = static_cast<float*>(buffer);
        for (size_t i = 0; i < elementCount; ++i)
        {
            result[i] = a[i] + b[i];
        }
    });

    // Set validation method
    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

    // Add tuning parameter: UNROLL_FACTOR with values 1, 2, 4, 8, 16
    tuner.AddParameter(kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8, 16});

    // Set launcher (just run kernel)
    tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
    {
        interface.RunKernel(definition);
    });

    // Tune with all configurations
    const auto results = tuner.Tune(kernel);

    // Print results
    std::cout << "\n=== Zero-Copy Tuning Results ===" << std::endl;
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

    // Verify the output buffer was modified by the kernel
    bool outputModified = false;
    for (size_t i = 0; i < elementCount; ++i)
    {
        if (out[i] != 0.0f)
        {
            outputModified = true;
            break;
        }
    }

    // User is responsible for freeing the memory
    free(a);
    free(b);
    free(out);

    if (allPassed && !results.empty() && outputModified)
    {
        std::cout << "\nZero-copy C++ backend tuning test passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nZero-copy C++ backend tuning test failed!" << std::endl;
        if (!outputModified)
        {
            std::cout << "Output buffer was not modified by kernel!" << std::endl;
        }
        return 1;
    }
}
