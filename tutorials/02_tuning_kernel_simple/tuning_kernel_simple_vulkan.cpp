#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../tutorials/02_tuning_kernel_simple/vulkan_kernel.glsl"
#else
    #define KTT_KERNEL_FILE "../../tutorials/02_tuning_kernel_simple/vulkan_kernel.glsl"
#endif

// Definition of class which will be used by tuner to automatically validate kernel output. It needs to publicly inherit from abstract class which is
// declared in KTT API.
class SimpleValidator : public ktt::ReferenceClass
{
public:
    // User-defined constructor. In this case it simply initializes all data that is needed to compute reference result.
    SimpleValidator(const ktt::ArgumentId validatedArgument, const std::vector<float>& a, const std::vector<float>& b,
        const std::vector<float>& result) :
        validatedArgument(validatedArgument),
        a(a),
        b(b),
        result(result)
    {}

    // Method inherited from ReferenceClass, which computes reference result for all arguments that are validated inside the class.
    void computeResult() override
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            result.at(i) = a.at(i) + b.at(i);
        }
    }

    // Method inherited from ReferenceClass, which returns memory location where reference result for corresponding argument is stored.
    void* getData(const ktt::ArgumentId id) override
    {
        if (validatedArgument == id)
        {
            return result.data();
        }
        return nullptr;
    }

private:
    ktt::ArgumentId validatedArgument;
    const std::vector<float>& a;
    const std::vector<float>& b;
    std::vector<float> result;
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
    // Work-group size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    const ktt::DimensionVector workGroupDimensions(1);
    const ktt::DimensionVector workGroupCount(numberOfElements);
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // Create new tuner for specified platform and device, tuner uses Vulkan as compute API.
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::Vulkan);

    // Add new kernel to tuner, specify path to kernel source, kernel function name, work-group count and dimensions. KTT returns handle to the newly
    // added kernel, which can be used to reference this kernel in other API methods.
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "main", workGroupCount, workGroupDimensions);

    // Add new kernel arguments to tuner, argument data is copied from std::vector containers. Specify whether the arguments are used as input
    // or output. KTT returns handle to the newly added arguemnts, which can be used to reference these arguments in other API methods. 
    ktt::ArgumentId aId = tuner.addArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);

    // Set arguments for the added kernel by providing their ids. The order of ids needs to match the order of bindings inside Vulkan shader.
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    // Set previously defined reference class to the kernel. Provide ids of kernel arguments which will be validated by this class. Each time
    // the kernel is run, output for all specified kernel arguments is compared to reference output computed inside the class.
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(resultId, a, b, result), std::vector<ktt::ArgumentId>{resultId});

    // Add a new parameter for kernel. Specify parameter name and possible values for this parameter. When kernel is tuned, the parameter value
    // is added to kernel source as preprocessor definition, eg. for parameter value 32, it is added as "#define multiply_work_group_size 32".
    tuner.addParameter(kernelId, "multiply_work_group_size", std::vector<size_t>{32, 64, 128, 256});

    // In this case, the parameter also affects work-group size. This is specified with KTT enums, ModifierType specifies that parameter affects
    // work-group size of a kernel, ModifierAction specifies that work-group size is multiplied by value of the parameter, ModifierDimension
    // specifies that dimension X of work-group is affected by the parameter.
    // Previously, the work-group size of kernel was set to one. This simply means that the work-group size of kernel is controlled explicitly by
    // value of this parameter, eg. size of one is multiplied by 32, which means that result size is 32.
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size", ktt::ModifierAction::Multiply);

    // Previously added parameter affects work-group size of kernel. However, when work-group size is changed, number of work-groups has to be
    // modified as well, so that number of work-groups multiplied by work-group size remains constant. This means that another modifier which divides
    // number of work-groups has to be added.
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_work_group_size", ktt::ModifierAction::Divide);

    // Set time unit used during printing of shader duration. Default time unit is milliseconds, but since computation in this tutorial does not take
    // too long, microseconds are used instead.
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    // Start tuning for specified kernel. This generates multiple versions of the kernel based on provided tuning parameters and their values. In
    // this case, 4 different versions of kernel will be run due to provided kernel constraint.
    tuner.tuneKernel(kernelId);

    // Print kernel run duration for each version of kernel (also called kernel configuration) to standard output in verbose format. This will also
    // print the fastest configuration at the end.
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

    return 0;
}
