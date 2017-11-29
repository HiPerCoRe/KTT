#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

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
    // Initialize device index and path to kernel.
    size_t deviceIndex = 0;
    std::string kernelFile = "../tutorials/02_tuning_kernel_simple/cuda_kernel.cu";

    if (argc >= 2)
    {
        deviceIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            kernelFile = std::string(argv[2]);
        }
    }

    // Declare kernel parameters and data variables.
    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector gridDimensions(numberOfElements);
    // Block size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    const ktt::DimensionVector blockDimensions(1);
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // Create new tuner for specified device, tuner uses CUDA as compute API.
    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::Cuda);

    // Add new kernel to tuner, specify path to kernel source, kernel function name, grid dimensions and block dimensions.
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "vectorAddition", gridDimensions, blockDimensions);

    // Add new kernel arguments to tuner, argument data is copied from std::vector containers.
    ktt::ArgumentId aId = tuner.addArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);

    // Set arguments for the kernel by providing their ids. The order of ids needs to match the order of arguments inside CUDA kernel function.
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    // Set previously defined reference class to the kernel. Provide ids of kernel arguments which will be validated by this class. Each time
    // the kernel is run, output for all specified kernel arguments is compared to reference output computed inside the class.
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(resultId, a, b, result), std::vector<ktt::ArgumentId>{resultId});

    // Add new parameter for kernel. Specify parameter name and possible values for this parameter. When kernel is tuned, the parameter value
    // is added to kernel source as preprocessor definition, eg. for parameter value 32, it is added as "#define multiply_block_size 32".
    // In this case, the parameter also affects block size. This is specified with KTT enums, ThreadModifierType specifies that parameter
    // affects block size of a kernel, ThreadModifierAction specifies that block size is multiplied by value of the parameter, dimension
    // specifies that dimension X of thread block is affected by the parameter.
    // Previously, the block size of kernel was set to one. This simply means that the block size of kernel is controlled explicitly by
    // value of this parameter, eg. size of one is multiplied by 32, which means that result size is 32.
    tuner.addParameter(kernelId, "multiply_block_size", std::vector<size_t>{32, 64, 128, 256}, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);

    // Previously added parameter affects thread block size of kernel. However, when block size is changed, grid size has to be modified as well,
    // so that grid size multiplied by block size remains unchanged. This means that another parameter which affects grid size needs to be added.
    tuner.addParameter(kernelId, "divide_grid_size", std::vector<size_t>{32, 64, 128, 256}, ktt::ThreadModifierType::Global,
        ktt::ThreadModifierAction::Divide, ktt::Dimension::X);

    // Add constraint to ensure that only valid versions of kernel will be generated. Previously, two kernel parameters with 4 possible values each
    // were added for kernel. This means that there are 16 possible versions of kernel that can be run, one version for each combination of parameter
    // values. However, in this case only versions where the two parameters have same value are valid. This can be specified to tuner by using
    // constraint. Constraint function receives several values for different parameters and checks whether their combination is valid. This function
    // is then added to tuner together with names of parameters that have their values checked inside the function.
    auto multiplyEqualsDivide = [](std::vector<size_t> vector) {return vector.at(0) == vector.at(1);}; 
    tuner.addConstraint(kernelId, multiplyEqualsDivide, std::vector<std::string>{"multiply_block_size", "divide_grid_size"});

    // Start tuning for specified kernel. This generates multiple versions of the kernel based on provided tuning parameters and their values. In
    // this case, 4 different versions of kernel will be run due to provided kernel constraint.
    tuner.tuneKernel(kernelId);

    // Print kernel run duration for each version of kernel (also called kernel configuration) to standard output in verbose format. This will also
    // print the fastest configuration at the end.
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

    return 0;
}
