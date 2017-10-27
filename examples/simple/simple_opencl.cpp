#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

// Implementation of reference class interface for result validation
class SimpleReferenceClass : public ktt::ReferenceClass
{
public:
    SimpleReferenceClass(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& result,
        const ktt::ArgumentId resultId) :
        a(a),
        b(b),
        result(result),
        resultId(resultId)
    {}

    void computeResult() override
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            result.at(i) = a.at(i) + b.at(i);
        }
    }

    const void* getData(const ktt::ArgumentId id) const override
    {
        if (id == resultId)
        {
            return (void*)result.data();
        }
        return nullptr;
    }

private:
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
    ktt::ArgumentId resultId;
};

int main(int argc, char** argv)
{
    // Initialize platform index, device index and path to kernel
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    std::string kernelFile = "../examples/simple/simple_opencl_kernel.cl";

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

    // Declare kernel parameters
    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector ndRangeDimensions(numberOfElements);
    const ktt::DimensionVector workGroupDimensions(256);

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

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add new kernel to tuner, specify kernel name, NDRange dimensions and work-group dimensions
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "simpleKernel", ndRangeDimensions, workGroupDimensions);

    // Add new arguments to tuner, argument data is copied from std::vector containers
    ktt::ArgumentId aId = tuner.addArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);

    // Set kernel arguments by providing corresponding argument ids returned by addArgument() method, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    // Set reference class, which implements C++ version of kernel computation in order to validate results provided by kernel,
    // provide list of arguments which will be validated
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleReferenceClass>(a, b, result, resultId), std::vector<ktt::ArgumentId>{resultId});

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "simple_opencl_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
