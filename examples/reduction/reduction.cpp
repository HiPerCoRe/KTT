#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"
#include "reduction_reference.h"

#if defined(_MSC_VER)
    const std::string kernelFilePrefix = "";
#else
    const std::string kernelFilePrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/reduction/reduction_kernel.cu";
    const auto computeAPI = ktt::ComputeAPI::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/reduction/reduction_kernel.cl";
    const auto computeAPI = ktt::ComputeAPI::OpenCL;
#endif

#define RAPID_TEST 0
#define USE_PROFILING 0

// Those macros enlarge tuning space by adding denser values to tuning 
// parameters (USE_DENSE_TUNPAR == 1), and also adding wider ranges of tuning
// parameters (USE_WIDE_TUNPAR  == 1)
#define USE_DENSE_TUNPAR 0
#define USE_WIDE_TUNPAR 0

#include "reduction_tunable.h"

int main(int argc, char** argv)
{
    // Initialize platform and device index
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
    #if USE_PROFILING == 0
    const int n = 64*1024*1024;
    #else
    const int n = 64*1024*1024/4;
    #endif
    const int nAlloc = ((n+16-1)/16)*16; // padd to longest vector size
    std::vector<float> src(nAlloc, 0.0f);
    std::vector<float> dst(nAlloc, 0.0f);

    for (int i = 0; i < n; i++)
    {
        src[i] = 1000.0f*((float)rand()) / ((float)RAND_MAX);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeAPI);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    #if USE_PROFILING == 1
    printf("Executing with profiling switched ON.\n");
    tuner.setKernelProfiling(true);
    #endif

    // create kernel
    int nUp = ((n+512-1)/512)*512; // maximum WG size used in tuning parameters
    ktt::DimensionVector ndRangeDimensions(nUp);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "reduce", ndRangeDimensions, workGroupDimensions);

    // create input/output
    ktt::ArgumentId srcId = tuner.addArgumentVector(src, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId dstId = tuner.addArgumentVector(dst, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId nId = tuner.addArgumentScalar(n);
    int offset = 0;
    ktt::ArgumentId inOffsetId = tuner.addArgumentScalar(offset);
    ktt::ArgumentId outOffsetId = tuner.addArgumentScalar(offset);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{srcId, dstId, nId, inOffsetId, outOffsetId});

#if RAPID_TEST == 1
    tuner.persistArgument(srcId, true);
    tuner.persistArgument(dstId, true);
#endif

    // get number of compute units
    const ktt::DeviceInfo di = tuner.getCurrentDeviceInfo();
    std::cout << "Number of compute units: " << di.getMaxComputeUnits() << std::endl;
    size_t cus = di.getMaxComputeUnits();

#if USE_DENSE_TUNPAR == 0 && USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 128, 256, 512});
#else
    #if USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 96, 128, 160, 196, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512});
    #else
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 96, 128, 160, 196, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024});
    #endif
#endif
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "UNBOUNDED_WG", {0, 1});
#if USE_DENSE_TUNPAR == 0 && USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
#else
    #if USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 3, cus * 4, cus * 5, cus * 6, cus * 7, cus * 8, cus * 10, cus * 12, cus * 14, cus * 16});
    #else
    tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 3, cus * 4, cus * 5, cus * 6, cus * 7, cus * 8, cus * 10, cus * 12, cus * 14, cus * 16, cus * 20, cus * 24, cus * 28, cus * 32, cus * 40, cus * 48, cus * 56, cus * 64});
    #endif
#endif

    if (computeAPI == ktt::ComputeAPI::OpenCL)
    {
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2, 4, 8, 16});
    }
    else
    {
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2, 4});
    }

    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_SIZE", ktt::ModifierAction::Divide);
    tuner.addParameter(kernelId, "USE_ATOMICS", {0, 1});

    auto persistConstraint = [](const std::vector<size_t>& v) {return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0);};
    tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "WG_NUM"}, persistConstraint);
    auto persistentAtomic = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[0] == 0 && v[1] == 1);};
    tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "USE_ATOMICS"}, persistentAtomic);
    auto unboundedWG = [](const std::vector<size_t>& v) {return (!v[0] || v[1] >= 32);};
    tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "WORK_GROUP_SIZE_X"}, unboundedWG);

#if RAPID_TEST == 0
    tuner.setReferenceClass(kernelId, std::make_unique<ReferenceReduction>(src, dstId), std::vector<ktt::ArgumentId>{dstId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, (double)n*10000.0/10'000'000.0);
    tuner.setValidationRange(dstId, 1);
#endif

    tuner.setTuningManipulator(kernelId, std::make_unique<TunableReduction>(srcId, dstId, nId, inOffsetId, outOffsetId));
    
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "reduction_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
