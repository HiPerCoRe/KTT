#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../include/ktt.h"

// Implementation of reference class interface for result validation
class SimpleReferenceClass : public ktt::ReferenceClass
{
public:
    SimpleReferenceClass(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& result, const size_t resultArgumentId) :
        a(a),
        b(b),
        result(result),
        resultArgumentId(resultArgumentId)
    {}

    virtual void computeResult() override
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            result.at(i) = a.at(i) + b.at(i);
        }
    }

    virtual void* getData(const size_t argumentId) const override
    {
        if (argumentId == resultArgumentId)
        {
            return (void*)result.data();
        }
        throw std::runtime_error("No result available for specified argument id");
    }

    virtual ktt::ArgumentDataType getDataType(const size_t argumentId) const override
    {
        return ktt::ArgumentDataType::Float;
    }

    virtual size_t getDataSizeInBytes(const size_t argumentId) const override
    {
        return result.size() * sizeof(float);
    }

private:
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
    size_t resultArgumentId;
};

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

    // Declare kernel parameters
    const std::string kernelFile = std::string("../examples/simple/simple_kernel.cl");
    const size_t numberOfElements = 1024 * 1024;
    ktt::DimensionVector ndRangeDimensions(numberOfElements, 1, 1);
    ktt::DimensionVector workGroupDimensions(256, 1, 1);

    // Declare data variables
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add new kernel to tuner, specify kernel name, NDRange dimensions and work-group dimensions
    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("simpleKernel"), ndRangeDimensions, workGroupDimensions);

    // Add new arguments to tuner, argument data is copied from std::vector containers
    size_t aId = tuner.addArgument(a, ktt::ArgumentMemoryType::ReadOnly);
    size_t bId = tuner.addArgument(b, ktt::ArgumentMemoryType::ReadOnly);
    size_t resultId = tuner.addArgument(result, ktt::ArgumentMemoryType::WriteOnly);

    // Set kernel arguments by providing corresponding argument ids returned by addArgument() method, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<size_t>{ aId, bId, resultId });

    // Set reference class, which implements C++ version of kernel computation in order to validate results provided by kernel,
    // provide list of arguments which will be validated
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleReferenceClass>(a, b, result, resultId), std::vector<size_t>{ resultId });

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);
    return 0;
}
