// KTT tutorial demonstrating tuning of a kernel that is built from several kernel definitions.
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
        Beginning of code similar to KernelTuning
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

    const int imageWidth = 1024;
    const int imageHeight = 1024;
    const size_t imageSize = static_cast<size_t>(imageWidth) * imageHeight;
    const int blurRadius = 4;
    const ktt::DimensionVector ndRangeDimensions(imageSize);
    const ktt::DimensionVector workGroupDimensions;

    std::vector<float> input(imageSize);
    std::vector<float> temporary(imageSize, 0.0f);
    std::vector<float> result(imageSize, 0.0f);

    for (int y = 0; y < imageHeight; ++y)
    {
        for (int x = 0; x < imageWidth; ++x)
        {
            input[(y * imageWidth) + x] = static_cast<float>((x + y) % 32);
        }
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    /******************************************************
        End of similar code
    ******************************************************/

    // This tutorial splits the blur into two passes, somewhat artificially, to present a composite kernel (kernel using
    // multiple kernel definitions)
    // This also showcases the difference between kernel definition and kernel. The kernel definition contains the kernel function
    // compiled and called by KTT and its associated arguments, the kernel launches one or more definitions, potentially with additional 
    // CPU processing, and contains the tuning parameters, constraints, and thread modifiers.
    const ktt::KernelDefinitionId horizontalDefinition = tuner.AddKernelDefinitionFromFile("blurHorizontally", kernelFile,
        ndRangeDimensions, workGroupDimensions);
    const ktt::KernelDefinitionId verticalDefinition = tuner.AddKernelDefinitionFromFile("blurVertically", kernelFile,
        ndRangeDimensions, workGroupDimensions);

    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, ktt::ArgumentAccessType::ReadOnly);
    // Note the ReadWrite access type, the first pass writes and the second pass reads.
    const ktt::ArgumentId temporaryId = tuner.AddArgumentVector(temporary, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId imageWidthId = tuner.AddArgumentScalar(imageWidth);
    const ktt::ArgumentId imageHeightId = tuner.AddArgumentScalar(imageHeight);
    const ktt::ArgumentId blurRadiusId = tuner.AddArgumentScalar(blurRadius);

    tuner.SetArguments(horizontalDefinition, {inputId, temporaryId, imageWidthId, imageHeightId, blurRadiusId});
    tuner.SetArguments(verticalDefinition, {temporaryId, resultId, imageWidthId, imageHeightId, blurRadiusId});

    // A composite kernel needs a kernel launcher to determine which definition runs when. The launcher
    // in the KernelLauncher tutorial uses it to move data between the host and the device, while here all
    // arguments stay on the device, so the launcher only has to run the two definitions in order.
    const ktt::KernelId kernel = tuner.CreateCompositeKernel("Blur", {horizontalDefinition, verticalDefinition},
        [horizontalDefinition, verticalDefinition](ktt::ComputeInterface& interface)
    {
        interface.RunKernel(horizontalDefinition);
        interface.RunKernel(verticalDefinition);
    });

    tuner.SetReferenceComputation(resultId, [&input](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);
        std::vector<float> temporaryArray(input.size(), 0.0f);

        for (int y = 0; y < imageHeight; ++y)
        {
            for (int x = 0; x < imageWidth; ++x)
            {
                float sum = 0.0f;
                int blurSize = 0;

                for (int offset = -blurRadius; offset <= blurRadius; ++offset)
                {
                    const int neighborX = x + offset;

                    if (neighborX >= 0 && neighborX < imageWidth)
                    {
                        sum += input[(y * imageWidth) + neighborX];
                        ++blurSize;
                    }
                }

                temporaryArray[(y * imageWidth) + x] = sum / blurSize;
            }
        }

        for (int y = 0; y < imageHeight; ++y)
        {
            for (int x = 0; x < imageWidth; ++x)
            {
                float sum = 0.0f;
                int blurSize = 0;

                for (int offset = -blurRadius; offset <= blurRadius; ++offset)
                {
                    const int neighborY = y + offset;

                    if (neighborY >= 0 && neighborY < imageHeight)
                    {
                        sum += temporaryArray[(neighborY * imageWidth) + x];
                        ++blurSize;
                    }
                }

                resultArray[(y * imageWidth) + x] = sum / blurSize;
            }
        }
    });

    // If set to 1, the intermediate result will be saved as column-major.
    // No specified group means it is in the "default" group.
    tuner.AddParameter(kernel, "TRANSPOSE_INTERMEDIATE", std::vector<uint64_t>{0, 1});

    // The work group sizes of the two passes should not affect each other, so they are tuned independently.
    // An example of the resulting process:
    // 1. Find the best combination of parameter values in group h_kernel, the others are set to arbitrary values.
    // 2. Keep h_kernel values as the best found, tune v_kernel group, the others are set to arbitrary values.
    // 3. Keep h_kernel and v_kernel values as the best found, tune the "default" group.
    // Splitting the tuning space like this means that the amount of configurations to explore is decreased greatly
    // (2+3+3 << 2*3*3), but we must be sure the parameters are really independent.
    tuner.AddParameter(kernel, "WG_SIZE_HORIZONTAL", std::vector<uint64_t>{32, 64, 128}, "h_kernel");
    tuner.AddParameter(kernel, "WG_SIZE_VERTICAL", std::vector<uint64_t>{32, 64, 128}, "v_kernel");

    tuner.AddThreadModifier(kernel, {horizontalDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
        "WG_SIZE_HORIZONTAL", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {verticalDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
        "WG_SIZE_VERTICAL", ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);

    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);
    return 0;
}
