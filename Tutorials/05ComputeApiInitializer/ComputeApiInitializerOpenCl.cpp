#include <iostream>
#include <string>
#include <vector>
#include <Cl/cl.h>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/05ComputeApiInitializer/OpenClKernel.cl";

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
    const ktt::DimensionVector workGroupDimensions;

    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }
    
    // Initialize OpenCL structures, no error checking is done to keep the code simple.
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, nullptr);

    // Using CL_QUEUE_PROFILING_ENABLE flag is mandatory if the queue is going to be used with the tuner.
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, nullptr);

    const size_t bufferSize = a.size() * sizeof(float);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, bufferSize, a.data(), 0, nullptr, nullptr);

    // Create compute API initializer which specifies context and streams that will be utilized by the tuner.
    ktt::ComputeApiInitializer initializer(context, std::vector<ktt::ComputeQueue>{queue});
    auto tunerUnique = std::make_unique<ktt::Tuner>(ktt::ComputeApi::OpenCL, initializer);

    // Utilize the tuner in the same way as in previous tutorials.
    auto& tuner = *tunerUnique;

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);

    // Add user-created buffer to tuner by providing its handle and size in bytes.
    const ktt::ArgumentId aId = tuner.AddArgumentVector<float>(bufferA, bufferSize, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    tuner.SetArguments(definition, {aId, bId, resultId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    tuner.SetReferenceComputation(resultId, [&a, &b](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < a.size(); ++i)
        {
            resultArray[i] = a[i] + b[i];
        }
    });

    tuner.AddParameter(kernel, "multiply_work_group_size", std::vector<uint64_t>{32, 64, 128, 256});
    tuner.SetThreadModifier(kernel, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size", {definition},
        ktt::ModifierAction::Multiply);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    // Make sure to delete the tuner before releasing referenced OpenCL structures.
    tunerUnique.reset();

    clReleaseMemObject(bufferA);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
