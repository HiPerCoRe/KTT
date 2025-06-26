/******************************************************************************
 * This is a set of microbenchmarks -- they stress different components of 
 * a GPU
 * The reason of existence of this example is testing speed and power 
 * consumption of used GPU.
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Microbenchmarks/Microbenchmarks.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Dummy/Microbenchmarks.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle kernel profiling.
const bool useProfiling = false;

const ktt::KernelResult& getBestResult(const std::vector<ktt::KernelResult>& res) {
    ktt::Nanoseconds bestTime = res[0].GetKernelDuration();
    int bestIdx = 0;
    for (int i = 0; i < res.size(); i++)
        if (res[i].GetKernelDuration() < bestTime) {
            bestTime = res[i].GetKernelDuration();
            bestIdx = i;
        }
    return res[bestIdx];
}

void reportKernelStats(ktt::KernelResult& bestTime, long long int sumTransferred){
    std::cout << "Configuration: "
        << bestTime.GetConfiguration().GetString() << "\n"
        << "Time, energy, and power: "
        << bestTime.GetKernelDuration() << " ns, "
        << bestTime.GetResults()[0].GetEnergyConsumption() << " J, "
        << bestTime.GetResults()[0].GetPowerUsage()/1000.0 << " W\n"
        << "Performance: " << sumTransferred / bestTime.GetKernelDuration()
        << "GB/s\n"
        << "Cost of transferring one byte: "
        << 1000000000000.0*bestTime.GetResults()[0].GetEnergyConsumption()/sumTransferred << " pJ\n";
}

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;

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

    // Declare and initialize data
    const int kernelSize = 1024*1024*32;
    const int touchedDataSize = 1024*1024*1024; // memory footprint
    const int touchedDataL2 = 1024*1024;
    const int touchedDataL1 = 16*1024;
    const long long int copyDataSize = (long long int)(1024*1024*1024)*100; 
    const int repeats = copyDataSize/kernelSize;

    const long long int sumTransferred = (long long int)(kernelSize) * (long long int)(repeats+1) * 4; //XXX +1 for writes

    const ktt::DimensionVector ndRangeDimensions(kernelSize);
    const ktt::DimensionVector workGroupDimensions(1);

    std::vector<int> input(touchedDataSize/4);
    std::vector<int> output(std::max(touchedDataSize/4, kernelSize));

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    tuner.SetLoggingLevel(ktt::LoggingLevel::Warning);
    if constexpr (computeApi == ktt::ComputeApi::CUDA)
    {
        if constexpr (useProfiling)
        {
            printf("Executing with profiling switched ON.\n");
            tuner.SetProfiling(true);
        }
    }


    const ktt::KernelDefinitionId defMemstress = tuner.AddKernelDefinitionFromFile("stressMem", kernelFile, ndRangeDimensions, workGroupDimensions);

    const ktt::KernelId kernelMemstress = tuner.CreateSimpleKernel("StressMem", defMemstress);

    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, 
        ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId outputId = tuner.AddArgumentVector(output, 
        ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId sizeId = tuner.AddArgumentScalar(touchedDataSize/4);
    const ktt::ArgumentId repeatsId = tuner.AddArgumentScalar(repeats);

    tuner.AddParameter(kernelMemstress, "BLOCK", 
        std::vector<uint64_t>{64, 128, 256, 512, 1024});
    tuner.AddThreadModifier(kernelMemstress, {defMemstress},
        ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::Multiply);

    tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeId, repeatsId});

    std::cout << "\nExecuting memory stress test: transferring "
        << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
        << touchedDataSize/1024/1024/1024 << "GB (to exceed L2)...";
    fflush(stdout);
    const auto results = tuner.Tune(kernelMemstress/*, std::make_unique<ktt::ConfigurationCount>(1)*/);
    tuner.SaveResults(results, "MemStressOutputGlobal", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "MemStressOutputGlobal", ktt::OutputFormat::XML);

    ktt::KernelResult bestTime = getBestResult(results);
    std::cout << "done.\n";
    reportKernelStats(bestTime, sumTransferred);

    std::cout << "\nExecuting memory stress test: transferring "
        << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
        << touchedDataL2/1024/1024 << "MB (to fit L2 but exceed L1)...";
    fflush(stdout);
    const ktt::KernelId kernelMemstressL2 = tuner.CreateSimpleKernel("StressMemL2", defMemstress);
    tuner.AddParameter(kernelMemstressL2, "BLOCK",
        std::vector<uint64_t>{64, 128, 256, 512, 1024});
    tuner.AddThreadModifier(kernelMemstressL2, {defMemstress},
        ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::Multiply);
    const ktt::ArgumentId sizeL2Id = tuner.AddArgumentScalar(touchedDataL2/4);
    tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeL2Id, repeatsId});

    const auto resultsL2 = tuner.Tune(kernelMemstressL2);
    tuner.SaveResults(results, "MemStressOutputL2", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "MemStressOutputL2", ktt::OutputFormat::XML);
    ktt::KernelResult bestTimeL2 = getBestResult(resultsL2);
    std::cout << "done.\n";
    reportKernelStats(bestTimeL2, sumTransferred);

    std::cout << "\nExecuting memory stress test: transferring "
        << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
        << touchedDataL1/1024 << "KB (to fit L1)...";
    fflush(stdout);
    const ktt::KernelId kernelMemstressL1 = tuner.CreateSimpleKernel("StressMemL1", defMemstress);
    tuner.AddParameter(kernelMemstressL1, "BLOCK",
        std::vector<uint64_t>{64, 128, 256, 512, 1024});
    tuner.AddThreadModifier(kernelMemstressL1, {defMemstress},
        ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::Multiply);
    const ktt::ArgumentId sizeL1Id = tuner.AddArgumentScalar(touchedDataL1/4);
    tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeL1Id, repeatsId});

    const auto resultsL1 = tuner.Tune(kernelMemstressL1);
    tuner.SaveResults(results, "MemStressOutputL1", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "MemStressOutputL1", ktt::OutputFormat::XML);
    ktt::KernelResult bestTimeL1 = getBestResult(resultsL1);
    std::cout << "done.\n";
    reportKernelStats(bestTimeL1, sumTransferred);


    return 0;
}
