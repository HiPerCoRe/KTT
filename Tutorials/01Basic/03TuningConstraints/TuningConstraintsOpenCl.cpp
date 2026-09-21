// KTT tutorial demonstrating the use of tuning constraints.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <cstdint>
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
    /******************************************************
        Beginning of code similar to KernelTuning
    ******************************************************/
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/01Basic/03TuningConstraints/OpenClKernel.cl";

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

    const int matrixSize = 1024;
    const size_t numberOfElements = static_cast<size_t>(matrixSize) * matrixSize;
    const ktt::DimensionVector ndRangeDimensions(matrixSize, matrixSize);
    const ktt::DimensionVector workGroupDimensions;

    std::vector<float> input(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        input[i] = static_cast<float>(i);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("matrixTranspose", kernelFile, ndRangeDimensions,
        workGroupDimensions);

    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId sizeId = tuner.AddArgumentScalar(static_cast<int>(matrixSize));
    tuner.SetArguments(definition, {inputId, resultId, sizeId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Transpose", definition);

    tuner.SetReferenceComputation(resultId, [&input](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (int x = 0; x < matrixSize; ++x)
        {
            for (int y = 0; y < matrixSize; ++y)
            {
                resultArray[y + x * matrixSize] = input[x + y * matrixSize];
            }
        }
    });
    /******************************************************
        End of similar code
    ******************************************************/

    // Work-group size in each dimension will be controlled by tuning parameters with values 1, 2, 4, ..., 512. Without any
    // constraints this gives 100 combinations of work-group sizes, most of which are either impractical or clearly suboptimal.
    std::vector<uint64_t> paramValues(10);
    uint64_t exp = 1;
    for (uint64_t i = 0; i < paramValues.size(); ++i)
    {
        paramValues[i] = exp;
        exp *= 2;
    }

    tuner.AddParameter(kernel, "WG_SIZE_X", paramValues);
    tuner.AddParameter(kernel, "WG_SIZE_Y", paramValues);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WG_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WG_SIZE_Y",
        ktt::ModifierAction::Multiply);

    // Tuning constraints can be used to filter out configurations that would cause the kernel to fail or those that would definitely
    // be suboptimal. Here we make sure that a work-group is neither too small (suboptimal) nor too big (fails). The constraint
    // function receives values of the listed parameters in the given order and returns true if the configuration should be tested.
    tuner.AddConstraint(kernel, {"WG_SIZE_X", "WG_SIZE_Y"}, [](std::vector<uint64_t> args)
    {
        return 32 <= args[0] * args[1] && args[0] * args[1] <= 512;
    });

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
