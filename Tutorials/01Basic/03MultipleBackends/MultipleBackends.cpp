// KTT tutorial demonstrating that the same program can be used to tune
// kernels using multiple different compute APIs.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include "ComputeEngine/ComputeApi.h"
#include "ComputeEngine/GlobalSizeType.h"
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if defined(KTT_CUDA_TUTORIAL)
    const std::string defaultKernelFile = kernelPrefix + "../Tutorials/01Basic/03MultipleBackends/CudaKernel.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif defined(KTT_OPENCL_TUTORIAL)
    const std::string defaultKernelFile = kernelPrefix + "../Tutorials/01Basic/03MultipleBackends/OpenClKernel.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#elif defined(KTT_CPP_TUTORIAL)
    const std::string defaultKernelFile = kernelPrefix + "../Tutorials/01Basic/03MultipleBackends/CppKernel.cppkernel";
    const auto computeApi = ktt::ComputeApi::Cpp;
#else
    #error "Tutorial must be compiled with KTT_CUDA_TUTORIAL, KTT_OPENCL_TUTORIAL or KTT_CPP_TUTORIAL defined."
#endif

int main(int argc, char** argv)
{
    /******************************************************
        Beginning of code similar to KernelTuning
    ******************************************************/
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

    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector ndRangeDimensions(numberOfElements);
    const ktt::DimensionVector workGroupDimensions;
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    
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
    /******************************************************
        End of similar code
    ******************************************************/

    if (computeApi == ktt::ComputeApi::Cpp)
    {
        tuner.SetCompilerOptions("-march=native -fopenmp");
        tuner.AddParameter(kernel, "OMP_SCHEDULING", std::vector<uint64_t>{0, 1, 2});
        tuner.AddParameter(kernel, "OMP_SCHED_CHUNK", std::vector<uint64_t>{2, 4, 8, 16, 32, 64, 128});

        tuner.AddConstraint(kernel, {"OMP_SCHEDULING", "OMP_SCHED_CHUNK"}, 
            [](const std::vector<uint64_t>& vector) {
                return vector.at(0) == 2 || vector.at(1) == 2;
            }
        );
    }
    else 
    {
        tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);  // KTT's backend will convert to CUDA sizing automatically if needed
        tuner.AddParameter(kernel, "multiply_work_group_size", std::vector<uint64_t>{32, 64, 128, 256});

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size",
            ktt::ModifierAction::Multiply);
    }

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
