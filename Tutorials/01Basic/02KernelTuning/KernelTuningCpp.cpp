// KTT tutorial demonstrating kernel setup and tuning via KTT
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <iostream>
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
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::Cpp);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalarValue);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // Set reference computation for the result argument which will be used by the tuner to automatically validate kernel output.
    // The computation function receives buffer on input, where the reference result should be saved. The size of buffer corresponds
    // to the validated argument size.
    tuner.SetReferenceComputation(resultId, [&a, &b, &scalarValue](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < a.size(); ++i)
        {
            resultArray[i] = a[i] + b[i] + scalarValue;
        }
    });

    // Set the compiler options used to compile every configuration of the kernel. The OpenMP pragmas in the kernel file
    // need -fopenmp, otherwise they are ignored and the loop runs on a single thread. See MultipleBackends and
    // CompilerTuning for more about compiler options.
    tuner.SetCompilerOptions("-O3 -fopenmp");

    // Add new kernel parameter. Specify parameter name and possible values. When kernel is tuned, the parameter value is added
    // to the beginning of kernel source as preprocessor definition. E.g., for value of this parameter equal to 1, it is added
    // as "#define OMP_SCHEDULING 1".
    tuner.AddParameter(kernel, "OMP_SCHEDULING", std::vector<uint64_t>{0, 1});

    // C++ kernels are regular functions, so they have no work-group sizes and no thread modifiers. A tuning parameter can
    // still control them through the kernel source itself: this one is a compile-time switch, read by the #if in the kernel
    // file to choose the OpenMP schedule of the loop.

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Perform tuning for the specified kernel. This generates multiple versions of the kernel based on provided tuning parameters
    // and their values. In this case, 2 different versions of kernel will be run.
    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}