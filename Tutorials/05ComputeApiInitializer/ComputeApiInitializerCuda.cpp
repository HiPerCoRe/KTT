#include <iostream>
#include <string>
#include <vector>
#include <cuda.h>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/05ComputeApiInitializer/CudaKernel.cu";

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

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    // Initialize CUDA structures, no error checking is done to keep the code simple.
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

    // Create compute API initializer which specifies context and streams that will be utilized by the tuner.
    ktt::ComputeApiInitializer initializer(context, std::vector<ktt::ComputeQueue>{stream});
    auto tunerUnique = std::make_unique<ktt::Tuner>(ktt::ComputeApi::CUDA, initializer);

    // Utilize the tuner in the same way as in previous tutorials.
    auto& tuner = *tunerUnique;

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions,
        blockDimensions);

    // Add user-created buffer to tuner by providing its handle and size in bytes.
    const ktt::ArgumentId aId = tuner.AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(bufferA), bufferSize,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
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

    tuner.AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Divide);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    // Make sure to delete the tuner before releasing referenced CUDA structures.
    tunerUnique.reset();

    cuMemFree(bufferA);
    cuStreamDestroy(stream);
    cuCtxDestroy(context);

    return 0;
}
