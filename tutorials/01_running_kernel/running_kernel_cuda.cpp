#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

int main(int argc, char** argv)
{
    // Initialize device index and path to kernel.
    size_t deviceIndex = 0;
    std::string kernelFile = "../tutorials/01_running_kernel/cuda_kernel.cu";

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
    // Dimensions of block and grid are specified with KTT data structure DimensionVector. Only single dimension is utilized in this tutorial.
    // In general, DimensionVector supports up to three dimensions.
    const ktt::DimensionVector blockDimensions(256);
    const ktt::DimensionVector gridDimensions(numberOfElements / blockDimensions.getSizeX());
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // Create new tuner for specified device, tuner uses CUDA as compute API. Platform index is ignored in this case.
    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::Cuda);

    // Add new kernel to tuner, specify path to kernel source, kernel function name, grid dimensions and block dimensions. KTT returns handle
    // to the newly added kernel, which can be used to reference this kernel in other API methods.
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "vectorAddition", gridDimensions, blockDimensions);

    // Add new kernel arguments to tuner, argument data is copied from std::vector containers. Specify whether the arguments are used as input
    // or output. KTT returns handle to the newly added arguemnts, which can be used to reference these arguments in other API methods. 
    ktt::ArgumentId aId = tuner.addArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);

    // Set arguments for the added kernel by providing their ids. The order of ids needs to match the order of arguments inside CUDA kernel
    // function.
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    // Run the specified kernel. Second argument is related to kernel tuning and will be described in further tutorials, here it remains empty. Third
    // argument is used to retrieve the kernel output. For each kernel argument that is retrieved, one ArgumentOutputDescriptor structure is created.
    // Each of these structures contains id of argument which is retrieved and memory location where the argument data will be stored. Optionally, it
    // can also include number of bytes to be retrieved, if only portion of argument is needed. Here the data is simply stored back into result
    // argument which was created earlier. Note that the memory location size needs to be equal or greater than retrieved argument size.
    tuner.runKernel(kernelId, {}, std::vector<ktt::ArgumentOutputDescriptor>{ktt::ArgumentOutputDescriptor(resultId, result.data())});

    return 0;
}
