// KTT tutorial demonstrating kernel setup and tuning via KTT
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

int main(int argc, char** argv)
{
    /******************************************************
        Beginning of code identical to KernelTuning
    ******************************************************/
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
    // Work-group size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    const ktt::DimensionVector workGroupDimensions;
    
    std::vector<float> input(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        input[i] = static_cast<float>(i % 32);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    /******************************************************
        End of identical code
    ******************************************************/

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    
    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalarValue);
    tuner.SetArguments(definition, {inputId, resultId});

    // We first create the simple kernel and then set the custom launcher. The custom launcher is usually a lambda function
    // capturing the necessary data.
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Reduction", definition);
    // This kernel launcher makes each work group reduce its assigned region of the input without synchronization between
    // groups and then does the final aggregation on the CPU. 
    // One could also pass the output back to the input with interface.SwapArguments(...) and run the kernel repeatedly
    // until only one value remains. An (admittedly rather complicated) implementation of this alternative approach is 
    // in Examples/Reduction.
    tuner.SetLauncher(kernel, [inputId, definition, resultId](ktt::ComputeInterface& interface)
    {
        const size_t globalSize = interface.GetCurrentGlobalSize(definition).GetSizeX();
        const size_t localSize = interface.GetCurrentLocalSize(definition).GetSizeX();
        const size_t groupNum = globalSize/localSize;
        std::vector<float> resultBuffer(groupNum);

        interface.RunKernel(definition);

        // We must transfer data between CPU and GPU manually in the kernel launcher.
        interface.DownloadBuffer(resultId, resultBuffer.data(), groupNum * sizeof(float));
        double accumulator = 0;
        for (size_t i = 0; i < groupNum; ++i) 
        {
            accumulator += resultBuffer[i];
            resultBuffer[i] = 0;
        }
        resultBuffer[0] = accumulator;
        interface.UpdateBuffer(resultId, resultBuffer.data(), groupNum * sizeof(float));
    });

    tuner.SetReferenceComputation(resultId, [&input](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < input.size(); ++i)
        {
            resultArray[0] += input[i];
        }
    });

    tuner.AddParameter(kernel, "WG_SIZE", std::vector<uint64_t>{32, 64, 128, 256});

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WG_SIZE",
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);
    return 0;
}
