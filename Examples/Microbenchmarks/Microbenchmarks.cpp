/******************************************************************************
 * This is a set of microbenchmarks -- they stress different components of 
 * a GPU
 * The reason of existence of this example is testing speed and power 
 * consumption of used GPU.
 */

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

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
const bool useProfiling = true;

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
    const int kernelSize = 1024*1024*128;
    const int dataRealSize = 1024*1024*1024; // memory footprint
    const long long int copyDataSize = (long long int)(1024*1024*1024)*100; //XXX transfered data = dataRealSize*copyDataSize, data must be big, or we are measuring just kernel overhead otherwise
    const int repeats = copyDataSize/kernelSize;
    assert(copyDataSize%kernelSize == 0);

    const ktt::DimensionVector ndRangeDimensions(kernelSize);
    const ktt::DimensionVector workGroupDimensions(1);

    std::vector<int> input(dataRealSize/4);
    std::vector<int> output(dataRealSize/4);

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
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
    const ktt::ArgumentId sizeId = tuner.AddArgumentScalar(dataRealSize/4);
    const ktt::ArgumentId repeatsId = tuner.AddArgumentScalar(repeats);

    tuner.AddParameter(kernelMemstress, "BLOCK", 
        std::vector<uint64_t>{64, 128, 256, 512, 1024});
    /*tuner.AddParameter(kernelMemstress, "OPS_PER_THREAD", 
        std::vector<uint64_t>{1, 2, 4, 8, 16, 32});*/
    tuner.AddThreadModifier(kernelMemstress, {defMemstress},
        ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::Multiply);
    /*tuner.AddThreadModifier(kernelMemstress, {defMemstress}, 
        ktt::ModifierType::Global, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::DivideCeil);*/
    /*tuner.AddThreadModifier(kernelMemstress, {defMemstress},
        ktt::ModifierType::Global, ktt::ModifierDimension::X, "OPS_PER_THREAD",
        ktt::ModifierAction::DivideCeil);*/

    tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeId, repeatsId});

    tuner.SetSearcher(kernelMemstress, std::make_unique<ktt::DeterministicSearcher>());

    const auto results = tuner.Tune(kernelMemstress/*, std::make_unique<ktt::ConfigurationCount>(1)*/);
    tuner.SaveResults(results, "MemStressOutput", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "MemStressOutput", ktt::OutputFormat::XML);

    ktt::KernelResult bestTime = getBestResult(results);
    std::cout << "Mem stress test performed.\n";
    std::cout << "Fastest kernel conf: "
        << bestTime.GetConfiguration().GetString() << "\n"
        << "Fastest kernel time and energy: " 
        << bestTime.GetKernelDuration() << "ns, "
        << bestTime.GetResults()[0].GetEnergyConsumption()*1000 << "mj\n"
        << "performance: " << copyDataSize / bestTime.GetKernelDuration() 
        << "GB/s\n";

    return 0;
}
