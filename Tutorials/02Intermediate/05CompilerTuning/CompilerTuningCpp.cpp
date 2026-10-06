// KTT tutorial demonstrating that the same program can be used to tune
// kernels using multiple different compute APIs.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

const std::string defaultKernelFile = kernelPrefix + KTT_TUTORIAL_KERNEL_FILE;

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

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::Cpp);

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

    // Taken from Cpp section of MultipleBackends
    tuner.SetCompilerOptions("-fopenmp");
    tuner.AddParameter(kernel, "OMP_SCHEDULING", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "OMP_SCHED_CHUNK", std::vector<uint64_t>{2, 4, 8, 16, 32, 64, 128});

    tuner.AddConstraint(kernel, {"OMP_SCHEDULING", "OMP_SCHED_CHUNK"}, 
        [](const std::vector<uint64_t>& vector) {
            return vector.at(0) == 2 || vector.at(1) == 2;
        }
    );

    // Tuning compiler arguments is sometimes worthwhile. For example, some C++ code can be faster under
    // -O2 than under -O3 and it is hard to tell without trying.

    // This parameter is tuned with the rest. 
    tuner.AddCompilerParameter(kernel, "-O", {"1", "2", "3"});
    // No supplied arguments mean that it is treated as a flag and either gets included in the compile command 
    // arguments or not.
    tuner.AddCompilerParameter(kernel, "-march=native");

    // Separate compiler tuning takes a base configuration tunes only the separate parameters. This works similarly 
    // to groups, but gives more control to the user. It may be useful for fine-tuning the best configuration without making 
    // the tuning space too large.
    tuner.AddSeparateCompilerParameter(kernel, "-ffast-math");
    tuner.AddSeparateCompilerParameter(kernel, "-fno-math-errno");

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> baseResults = tuner.Tune(kernel);

    // Separate compiler tuning must be started manually and also involves choosing the configuration to fine-tune.
    // This also means that one could choose several promising configurations for fine-tuning.
    auto bestConfiguration = tuner.GetBestConfiguration(kernel);
    const auto optionsResults = tuner.TuneOptions(kernel, bestConfiguration);

    tuner.SaveResults(baseResults, "TuningOutputBase", ktt::OutputFormat::JSON);
    tuner.SaveResults(optionsResults, "TuningOutputOptions", ktt::OutputFormat::JSON);

    return 0;
}
