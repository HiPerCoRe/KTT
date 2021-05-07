#include <iostream>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/06VectorArgumentCustomization/CudaKernel.cu";

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

    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions,
        blockDimensions);

    // Argument memory location specifies the memory from which kernel accesses data of the corresponding buffer. You may try
    // changing this setting and observing how it affects kernel computation times (e.g., for Nvidia GPUs, using device memory
    // is usually going to be significantly faster than host memory).
    constexpr ktt::ArgumentMemoryLocation location = ktt::ArgumentMemoryLocation::Device;

    // Argument management type specifies who is responsible for uploading and clearing compute API buffer corresponding to the
    // specified argument. If it is framework, the buffer will be handled automatically. If it is user, the buffer must be managed
    // manually inside kernel launcher.
    constexpr ktt::ArgumentManagementType management = ktt::ArgumentManagementType::User;

    // Reference user data flag specifies whether the argument will be copied into internal KTT buffer or accessed directly from
    // user-provided buffer (in this case std::vector a and b). Setting this flag to true reduces memory usage but requires user
    // to keep the original buffer valid for the entire duration of tuning. That is easily achieved in this tutorial.
    constexpr bool referenceData = true;

    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly, location, management, referenceData);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly, location, management, referenceData);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly, location,
        ktt::ArgumentManagementType::Framework, false);
    tuner.SetArguments(definition, {aId, bId, resultId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    if constexpr (management == ktt::ArgumentManagementType::User)
    {
        // User is responsible for buffer management in this case. That means uploading the corresponding buffers before kernel is
        // run and clearing them afterwards. Note that if the given buffer is a result buffer which is validated, it must not be
        // cleared right after the kernel run is finished. It should be cleared during the next launcher invocation. The buffer
        // handling is counted as overhead during measurement of computation duration.
        tuner.SetLauncher(kernel, [kernel, aId, bId](ktt::ComputeInterface& interface)
        {
            interface.UploadBuffer(aId);
            interface.UploadBuffer(bId);

            interface.RunKernel(kernel);

            interface.ClearBuffer(aId);
            interface.ClearBuffer(bId);
        });
    }

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

    const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
