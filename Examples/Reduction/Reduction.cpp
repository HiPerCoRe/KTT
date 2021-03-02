#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Reduction/Reduction.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Reduction/Reduction.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Add denser values to tuning parameters (useDenseParameters = true).
const bool useDenseParameters = false;

// Add wider ranges of tuning parameters (useWideParameters  = true).
const bool useWideParameters = false;

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
    uint32_t n;

    if constexpr (!useProfiling)
    {
        n = 64 * 1024 * 1024;
    }
    else
    {
        n = 64 * 1024 * 1024 / 4;
    }

    const uint32_t nAlloc = ((n+16-1)/16)*16; // pad to the longest vector size
    std::vector<float> src(nAlloc, 0.0f);
    std::vector<float> dst(nAlloc, 0.0f);

    for (uint32_t i = 0; i < n; ++i)
    {
        src[i] = 1000.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    const uint32_t nUp = ((n+512-1)/512)*512; // maximum WG size used in tuning parameters
    ktt::DimensionVector ndRangeDimensions(nUp);
    ktt::DimensionVector workGroupDimensions;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, ndRangeDimensions,
        workGroupDimensions);

    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId dstId = tuner.AddArgumentVector(dst, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(n);
    uint32_t offset = 0;
    const ktt::ArgumentId inOffsetId = tuner.AddArgumentScalar(offset);
    const ktt::ArgumentId outOffsetId = tuner.AddArgumentScalar(offset);
    tuner.SetArguments(definition, {srcId, dstId, nId, inOffsetId, outOffsetId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Reduction", definition);

    // get number of compute units
    const ktt::DeviceInfo di = tuner.GetCurrentDeviceInfo();
    std::cout << "Number of compute units: " << di.GetMaxComputeUnits() << std::endl;
    size_t cus = di.GetMaxComputeUnits();

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{32, 64, 128, 256, 512});
    }
    else if constexpr (!useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{32, 64, 96, 128, 160, 196, 224, 256, 288, 320, 352,
            384, 416, 448, 480, 512});
    }
    else
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{32, 64, 96, 128, 160, 196, 224, 256, 288, 320, 352,
            384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024});
    }

    tuner.SetThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddParameter(kernel, "UNBOUNDED_WG", std::vector<uint64_t>{0, 1});

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "WG_NUM", std::vector<uint64_t>{0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
    }
    else if constexpr (!useWideParameters)
    {
        tuner.AddParameter(kernel, "WG_NUM", std::vector<uint64_t>{0, cus, cus * 2, cus * 3, cus * 4, cus * 5, cus * 6, cus * 7,
            cus * 8, cus * 10, cus * 12, cus * 14, cus * 16});
    }
    else
    {
        tuner.AddParameter(kernel, "WG_NUM", std::vector<uint64_t>{0, cus, cus * 2, cus * 3, cus * 4, cus * 5, cus * 6, cus * 7,
            cus * 8, cus * 10, cus * 12, cus * 14, cus * 16, cus * 20, cus * 24, cus * 28, cus * 32, cus * 40, cus * 48, cus * 56, cus * 64});
    }

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1, 2, 4, 8, 16});
    }
    else
    {
        tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1, 2, 4});
    }

    tuner.SetThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_SIZE",
        ktt::ModifierAction::Divide);
    tuner.AddParameter(kernel, "USE_ATOMICS", std::vector<uint64_t>{0, 1});

    auto persistConstraint = [](const std::vector<uint64_t>& v) {return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0);};
    tuner.AddConstraint(kernel, {"UNBOUNDED_WG", "WG_NUM"}, persistConstraint);
    auto persistentAtomic = [](const std::vector<uint64_t>& v) {return (v[0] == 1) || (v[0] == 0 && v[1] == 1);};
    tuner.AddConstraint(kernel, {"UNBOUNDED_WG", "USE_ATOMICS"}, persistentAtomic);
    auto unboundedWG = [](const std::vector<uint64_t>& v) {return (!v[0] || v[1] >= 32);};
    tuner.AddConstraint(kernel, {"UNBOUNDED_WG", "WORK_GROUP_SIZE_X"}, unboundedWG);

    if constexpr (!rapidTest)
    {
        tuner.SetReferenceComputation(dstId, [&src](void* buffer)
        {
            float* result = static_cast<float*>(buffer);
            std::vector<double> resD(src.size());
            size_t resSize = src.size();

            for (size_t i = 0; i < resSize; ++i)
            {
                resD[i] = static_cast<double>(src[i]);
            }

            while (resSize > 1)
            {
                for (size_t i = 0; i < resSize / 2; ++i)
                {
                    resD[i] = resD[i * 2] + resD[i * 2 + 1];
                }

                if (resSize % 2 != 0)
                {
                    resD[resSize / 2 - 1] += resD[resSize - 1];
                }

                resSize = resSize / 2;
            }

            std::cout << "Reference in double: " << std::setprecision(10) << resD[0] << std::endl;
            result[0] = static_cast<float>(resD[0]);
        });

        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, static_cast<double>(n) * 10'000.0 / 10'000'000.0);
        tuner.SetValidationRange(dstId, 1);
    }

    tuner.SetLauncher(kernel, [definition, srcId, dstId, nId, inOffsetId, outOffsetId](ktt::ComputeInterface& interface)
    {
        const ktt::DimensionVector& globalSize = interface.GetCurrentGlobalSize(definition);
        const ktt::DimensionVector& localSize = interface.GetCurrentLocalSize(definition);
        const std::vector<ktt::ParameterPair>& pairs = interface.GetCurrentConfiguration().GetPairs();
        ktt::DimensionVector myGlobalSize = globalSize;

        // change global size for constant numbers of work-groups
        // this may be done by thread modifier operators as well
        if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "UNBOUNDED_WG") == 0)
        {
            myGlobalSize = ktt::DimensionVector(ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "WG_NUM")
                * localSize.GetSizeX());
        }

        // execute reduction kernel
        if constexpr (!useProfiling)
        {
            interface.RunKernel(definition, myGlobalSize, localSize);
        }
        else
        {
            interface.RunKernelWithProfiling(definition, myGlobalSize, localSize);
        }

        // execute kernel log n times, when atomics are not used 
        if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "USE_ATOMICS") == 0)
        {
            uint32_t n = static_cast<uint32_t>(globalSize.GetSizeX() / localSize.GetSizeX());
            uint32_t inOffset = 0;
            uint32_t outOffset = n;
            uint32_t vectorSize = static_cast<uint32_t>(ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "VECTOR_SIZE"));
            uint32_t wgSize = static_cast<uint32_t>(localSize.GetSizeX());
            size_t iterations = 0; // make sure the end result is in the correct buffer

            while (n > 1 || iterations % 2 == 1)
            {
                interface.SwapArguments(definition, srcId, dstId);
                myGlobalSize.SetSizeX((n + vectorSize - 1) / vectorSize);
                myGlobalSize.SetSizeX(((myGlobalSize.GetSizeX() - 1) / wgSize + 1) * wgSize);
                
                if (myGlobalSize == localSize)
                {
                    outOffset = 0; // only one WG will be executed
                }

                interface.UpdateScalarArgument(nId, &n);
                interface.UpdateScalarArgument(outOffsetId, &outOffset);
                interface.UpdateScalarArgument(inOffsetId, &inOffset);

                interface.RunKernel(definition, myGlobalSize, localSize);
                n = (n + wgSize * vectorSize - 1) / (wgSize * vectorSize);
                inOffset = outOffset / vectorSize; // input is vectorized, output is scalar
                outOffset += n;
                ++iterations;
            }
        }
    });
    
    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "ReductionOutput", ktt::OutputFormat::JSON);

    return 0;
}
