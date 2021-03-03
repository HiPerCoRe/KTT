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
    std::string kernelFile = kernelPrefix + "../Tutorials/03KernelTuning/CudaKernel.cu";

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
    // Block size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    const ktt::DimensionVector blockDimensions;
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions,
        blockDimensions);
    
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalarValue);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // Set reference computation for the result argument which will be used by the tuner to automatically validate kernel output.
    // The computation function receives buffer on input, where the reference result should be saved. The size of buffer corresponds
    // to the validated argument size.
    tuner.SetReferenceComputation(resultId, [&a, &b, scalarValue](void* buffer)
    {
        float* resultArray = static_cast<float*>(buffer);

        for (size_t i = 0; i < a.size(); ++i)
        {
            resultArray[i] = a[i] + b[i] + scalarValue;
        }
    });

    // Add new kernel parameter. Specify parameter name and possible values. When kernel is tuned, the parameter value is added
    // to the beginning of kernel source as preprocessor definition. E.g., for value of this parameter equal to 32, it is added
    // as "#define multiply_block_size 32".
    tuner.AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256});

    // In this case, the parameter also affects block size. This is specified by adding a thread modifier. ModifierType specifies
    // that parameter affects block size of a kernel, ModifierAction specifies that block size is multiplied by value of the
    // parameter, ModifierDimension specifies that dimension X of a thread block is affected by the parameter. It is also possible
    // to specify which definitions are affected by the modifier. In this case, only one definition is affected. The default block
    // size inside kernel definition was set to one. This means that the block size of the definition is controlled explicitly by
    // value of this parameter. E.g., size of one is multiplied by 32, which means that result size is 32.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Multiply);

    // Previously added parameter affects thread block size of kernel. However, when block size is changed, grid size has to be
    // modified as well, so that grid size multiplied by block size remains constant. This means that another modifier which divides
    // grid size has to be added.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Divide);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Perform tuning for the specified kernel. This generates multiple versions of the kernel based on provided tuning parameters
    // and their values. In this case, 4 different versions of kernel will be run.
    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel);

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);

    return 0;
}
