#include <iostream>
#include <vector>
#include <random>
#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

const std::string defaultKernelFile = kernelPrefix + "../Examples/CppTranspose/CppTransposeKernel.cppkernel";
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

    // Set compiler options for optimization and OpenMP support
    tuner.SetCompilerOptions("-O3 -march=native -fopenmp");

    // Define matrix dimensions
    const size_t matrixWidth = 1024*16;
    const size_t matrixHeight = 1024*16;
    const size_t elementCount = matrixWidth * matrixHeight;
    
    // Kernel dimensions (1D work size for simplicity)
    const ktt::DimensionVector ndRangeDimensions(1);
    const ktt::DimensionVector workGroupDimensions(1);

    // Kernel definition from file
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile(
        "transpose",
        kernelFile,
        ndRangeDimensions,
        workGroupDimensions
    );

    // Create simple kernel
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("MatrixTranspose", definition);

    // Create matrices
    std::vector<float> inputMatrix(elementCount);
    std::vector<float> outputMatrix(elementCount, 0.0f);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < elementCount; ++i)
    {
        inputMatrix[i] = dist(gen);
    }

    // Add arguments
    // Output matrix (transposed)
    const ktt::ArgumentId argOutput = tuner.AddArgumentVector<float>(outputMatrix.data(), outputMatrix.size()*sizeof(outputMatrix[0]), ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Host);
    // Input matrix
    const ktt::ArgumentId argInput = tuner.AddArgumentVector<float>(inputMatrix.data(), inputMatrix.size()*sizeof(inputMatrix[0]), ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Host);
    // Width parameter (scalar)
    const ktt::ArgumentId argWidth = tuner.AddArgumentScalar(matrixWidth);
    // Height parameter (scalar)
    const ktt::ArgumentId argHeight = tuner.AddArgumentScalar(matrixHeight);

    // Associate arguments with kernel definition
    tuner.SetArguments(definition, {argOutput, argInput, argWidth, argHeight});

    // Set reference computation for validation
    tuner.SetReferenceComputation(argOutput, [&inputMatrix, matrixWidth, matrixHeight](void* buffer)
    {
        float* result = static_cast<float*>(buffer);
        // Reference implementation: naive matrix transposition
        for (size_t y = 0; y < matrixHeight; ++y)
        {
            for (size_t x = 0; x < matrixWidth; ++x)
            {
                // Transpose: result[x][y] = input[y][x]
                result[x * matrixHeight + y] = inputMatrix[y * matrixWidth + x];
            }
        }
    });

    // Set validation method
    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

    // Add tuning parameter: TILE_SIZE with various values
    // Different tile sizes affect cache locality and performance
    tuner.AddParameter(kernel, "TILE_SIZE", std::vector<uint64_t>{4, 8, 16, 32, 64, 128, 256, 512});
    tuner.AddParameter(kernel, "OMP_COLLAPSE", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "OMP_SCHEDULING", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "OMP_SCHED_CHUNK", std::vector<uint64_t>{2, 4, 8, 16, 32, 64, 128});

    // Define constraint, that sets OMP_SCHED_CHUNK=2 always when OMP_SCHEDULING != 2 
    // (as for OMP_SCHEDULING = 0 or 1, OMP_SCHED_CHUNK has no effect)
    auto schedConstraint = [] (const std::vector<uint64_t>& v) { return ((v[0] == 2) || (v[1] == 2)); };
    tuner.AddConstraint(kernel, {"OMP_SCHEDULING", "OMP_SCHED_CHUNK"}, schedConstraint);

    // Tune with all configurations
    std::cout << "Starting matrix transpose tuning..." << std::endl;
    std::cout << "Matrix size: " << matrixWidth << " x " << matrixHeight << std::endl;
    std::cout << "Testing different tile sizes for cache optimization..." << std::endl;
    std::cout << "Compiler options: -O3 -march=native -fopenmp" << std::endl;
    
    const auto results = tuner.Tune(kernel);

    // Print results
    std::cout << "\n=== Matrix Transpose Tuning Results ===" << std::endl;
    std::cout << "Total configurations tested: " << results.size() << std::endl;
    
    bool allPassed = true;
    double bestDuration = std::numeric_limits<double>::max();
    std::string bestConfig;
    
    for (const auto& result : results)
    {
        double duration = result.GetTotalDuration();
        std::string config = result.GetConfiguration().GetString();
        
        std::cout << "Configuration: " << config
                  << " -> Duration: " << duration << " ns"
                  << " -> Status: " << (result.GetStatus() == ktt::ResultStatus::Ok ? "OK" : "FAILED")
                  << std::endl;
        
        if (result.GetStatus() != ktt::ResultStatus::Ok)
        {
            allPassed = false;
        }
        else if (duration < bestDuration)
        {
            bestDuration = duration;
            bestConfig = config;
        }
    }

    tuner.SaveResults(results, "CppTranspose", ktt::OutputFormat::JSON);

    if (allPassed && !results.empty())
    {
        std::cout << "\nBest configuration: " << bestConfig << " with duration " << bestDuration << " ns (" 
            << (double)elementCount*2.0*(double)sizeof(inputMatrix[0])/(double)bestDuration << "GB/s)"<< std::endl;
        std::cout << "C++ backend matrix transpose tuning test passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nC++ backend matrix transpose tuning test failed!" << std::endl;
        return 1;
    }
}
