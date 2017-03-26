#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

int main(int argc, char** argv)
{
    // Initialize platform and device index
    size_t platformIndex = 0;
    size_t deviceIndex = 0;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string{ argv[1] });
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string{ argv[2] });
        }
    }

    // Declare constants
    const float upperBoundary = 1000.0f; // used for generating random test data
    const std::string kernelFile = std::string("simple_kernel.cl");
    const std::string referenceKernelFile = std::string("simple_reference_kernel.cl");

    // Declare kernel parameters
    const int numberOfElements = 512 * 512;
    ktt::DimensionVector ndRangeDimensions(numberOfElements, 1, 1);
    ktt::DimensionVector workGroupDimensions(256, 1, 1);

    // Declare data variables
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (int i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex);

    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("simpleKernel"), ndRangeDimensions, workGroupDimensions);
    size_t aId = tuner.addArgument(a, ktt::ArgumentMemoryType::READ_ONLY);
    size_t bId = tuner.addArgument(b, ktt::ArgumentMemoryType::READ_ONLY);
    size_t resultId = tuner.addArgument(result, ktt::ArgumentMemoryType::WRITE_ONLY);

    tuner.setKernelArguments(kernelId, std::vector<size_t>{ aId, bId, resultId });
    tuner.tuneKernel(kernelId);

    return 0;
}
