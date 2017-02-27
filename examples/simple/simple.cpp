#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
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
    const std::string kernelName = std::string("simple_kernel.cl");
    const std::string referenceKernelName = std::string("simple_reference_kernel.cl");

    // Declare kernel parameters
    const int numberOfElements = 4096 * 4096;
    ktt::DimensionVector ndRangeDimensions(4096 * 4096, 1, 1);
    ktt::DimensionVector workGroupDimensions(256, 1, 1);

    // Declare data variables
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    srand((unsigned)time(0));

    for (int i = 0; i < numberOfElements; i++)
    {
        a.at(i) = (float)rand() / (RAND_MAX / upperBoundary);
        b.at(i) = (float)rand() / (RAND_MAX / upperBoundary);
    }
    
    // WIP
    ktt::Tuner::printOpenCLInfo(std::cout);
    ktt::Tuner tuner(platformIndex, { deviceIndex });

    size_t kernelId = tuner.addKernelFromFile(kernelName, std::string("multirunKernel"), ndRangeDimensions, workGroupDimensions);
    tuner.addArgumentFloat(kernelId, a);
    tuner.addArgumentFloat(kernelId, b);
    tuner.addArgumentFloat(kernelId, result);

    tuner.useSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double>{ 0.5 });
    return 0;
}
