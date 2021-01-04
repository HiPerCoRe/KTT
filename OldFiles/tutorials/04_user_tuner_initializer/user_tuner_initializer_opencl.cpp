#include <iostream>
#include <string>
#include <vector>
#include <CL/cl.h>
#include "tuner_api.h"

#if defined(_MSC_VER)
    const std::string kernelFilePrefix = "";
#else
    const std::string kernelFilePrefix = "../";
#endif

class SimpleValidator : public ktt::ReferenceClass
{
public:
    SimpleValidator(const ktt::ArgumentId validatedArgument, const std::vector<float>& a, const std::vector<float>& b,
        const std::vector<float>& result) :
        validatedArgument(validatedArgument),
        a(a),
        b(b),
        result(result)
    {}

    void computeResult() override
    {
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void* getData(const ktt::ArgumentId /*id*/) override
    {
        return result.data();
    }

private:
    ktt::ArgumentId validatedArgument;
    const std::vector<float>& a;
    const std::vector<float>& b;
    std::vector<float> result;
};

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelFilePrefix + "../tutorials/04_user_tuner_initializer/kernel.cl";

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
    
    // User-initialized OpenCL structures, no error checking is done to keep the code simple.
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

    // Create user tuner initializer which specifies context and queues that will be utilized by the tuner.
    ktt::UserInitializer initializer(context, std::vector<ktt::UserQueue>{queue});
    auto tunerUnique = std::make_unique<ktt::Tuner>(ktt::ComputeAPI::OpenCL, initializer);

    // Utilize the tuner in the same way as in previous tutorials.
    auto& tuner = *tunerUnique;

    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "vectorAddition", ndRangeDimensions, workGroupDimensions);

    // Add user-created buffer to tuner by providing its handle and size in bytes.
    ktt::ArgumentId aId = tuner.addArgumentVector<float>(bufferA, bufferSize, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(resultId, a, b, result), std::vector<ktt::ArgumentId>{resultId});

    tuner.addParameter(kernelId, "multiply_work_group_size", std::vector<size_t>{32, 64, 128, 256});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_work_group_size", ktt::ModifierAction::Multiply);
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

    // Make sure to delete the tuner before releasing referenced OpenCL structures.
    tunerUnique.reset();

    clReleaseMemObject(bufferA);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
