#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../tutorials/03_custom_kernel_arguments/opencl_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../tutorials/03_custom_kernel_arguments/opencl_kernel.cl"
#endif

// Definition of our custom data type, it needs to be defined / visible in kernel source code as well.
struct KernelData
{
    float a;
    float b;
    float result;
};

// Function which checks that two different struct instances contain the same result. This is needed by tuner to validate kernel arguments with
// custom data types.
bool compareData(const void* resultPointer, const void* referencePointer)
{
    KernelData* result = (KernelData*)resultPointer;
    KernelData* reference = (KernelData*)referencePointer;

    if (result->result != reference->result)
    {
        std::cerr << "Result: " << result->result << ", reference: " << reference->result << std::endl;
        return false;
    }

    return true;
}

// Definition of reference class.
class SimpleValidator : public ktt::ReferenceClass
{
public:
    SimpleValidator(const ktt::ArgumentId validatedArgument, const std::vector<KernelData>& data) :
        validatedArgument(validatedArgument),
        data(data)
    {}

    void computeResult() override
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            data.at(i).result = data.at(i).a + data.at(i).b;
        }
    }

    void* getData(const ktt::ArgumentId id) override
    {
        if (validatedArgument == id)
        {
            return data.data();
        }
        return nullptr;
    }

private:
    ktt::ArgumentId validatedArgument;
    std::vector<KernelData> data;
};

int main(int argc, char** argv)
{
    // Initialize platform index, device index and path to kernel.
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;

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

    // Declare kernel parameters and data variables.
    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector ndRangeDimensions(numberOfElements);
    std::vector<KernelData> data(numberOfElements);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        data.at(i).a = static_cast<float>(i);
        data.at(i).b = static_cast<float>(i + 1);
        data.at(i).result = 0.0f;
    }

    // Create new tuner for specified platform and device, tuner uses OpenCL as compute API.
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add new kernel to tuner, specify path to kernel source, kernel function name, ND-range dimensions and work-group dimensions.
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "vectorAddition", ndRangeDimensions, workGroupDimensions);

    // Add kernel argument with custom data type to tuner.
    ktt::ArgumentId dataId = tuner.addArgumentVector(data, ktt::ArgumentAccessType::ReadWrite);

    // Set argument for the kernel.
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{dataId});

    // Set reference class for the kernel.
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(dataId, data), std::vector<ktt::ArgumentId>{dataId});

    // Set previously defined function as comparator for the kernel argument.
    tuner.setArgumentComparator(dataId, compareData);

    // Add parameter and thread modifier for the kernel. See previous tutorial for more information.
    tuner.addParameter(kernelId, "multiply_work_group_size", std::vector<size_t>{32, 64, 128, 256});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size", ktt::ModifierAction::Multiply);

    // Set time unit used during printing of kernel duration to microseconds.
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    // Start kernel tuning and print results.
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

    return 0;
}
