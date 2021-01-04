#include <iostream>
#include <string>
#include <vector>
#include <cuda.h>
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
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelFilePrefix + "../tutorials/04_user_tuner_initializer/kernel.cu";

    if (argc >= 2)
    {
        deviceIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            kernelFile = std::string(argv[2]);
        }
    }

    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector gridDimensions(numberOfElements);
    const ktt::DimensionVector blockDimensions;
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // User-initialized CUDA structures, no error checking is done to keep the code simple.
    cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);

    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    const size_t bufferSize = a.size() * sizeof(float);
    CUdeviceptr bufferA;
    cuMemAlloc(&bufferA, bufferSize);
    cuMemcpyHtoD(bufferA, a.data(), bufferSize);

    // Create user tuner initializer which specifies context and streams that will be utilized by the tuner.
    ktt::UserInitializer initializer(context, std::vector<ktt::UserQueue>{stream});
    auto tunerUnique = std::make_unique<ktt::Tuner>(ktt::ComputeAPI::CUDA, initializer);

    // Utilize the tuner in the same way as in previous tutorials.
    auto& tuner = *tunerUnique;

    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "vectorAddition", gridDimensions, blockDimensions);

    // Add user-created buffer to tuner by providing its handle and size in bytes.
    ktt::ArgumentId aId = tuner.addArgumentVector<float>(reinterpret_cast<ktt::UserBuffer>(bufferA), bufferSize, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    ktt::ArgumentId bId = tuner.addArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId resultId = tuner.addArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aId, bId, resultId});

    tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(resultId, a, b, result), std::vector<ktt::ArgumentId>{resultId});

    tuner.addParameter(kernelId, "multiply_block_size", std::vector<size_t>{32, 64, 128, 256});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size", ktt::ModifierAction::Divide);

    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

    // Make sure to delete the tuner before releasing referenced CUDA structures.
    tunerUnique.reset();

    cuMemFree(bufferA);
    cuStreamDestroy(stream);
    cuCtxDestroy(context);

    return 0;
}
