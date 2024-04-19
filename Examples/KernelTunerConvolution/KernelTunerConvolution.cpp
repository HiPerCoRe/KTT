#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/KernelTunerConvolution/KernelTunerConvolution.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/KernelTunerConvolution/KernelTunerConvolutionReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/KernelTunerConvolution/KernelTunerConvolution.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/KernelTunerConvolution/KernelTunerConvolutionReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Settings (synchronise these with kernel source files)
const int HFS = 7; // Half filter size
const int FS = (HFS + HFS + 1); // Filter size

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return ((a / b) * b == a) ? true : false;
}

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;
    std::string referenceKernelFile = defaultReferenceKernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);

                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // Declare data variables
    uint32_t kSizeX; // Matrix dimension X
    uint32_t kSizeY; // Matrix dimension Y

    if constexpr (!useProfiling)
    {
        kSizeX = 4096;
        kSizeY = 4096;
    }
    else
    {
        kSizeX = 4096;
        kSizeY = 4096;
    }

    const ktt::DimensionVector ndRangeDimensions(kSizeX, kSizeY);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    std::vector<float> input((kSizeX + 2 * HFS) * (kSizeY + 2 * HFS), 0.0f);
    std::vector<float> output(kSizeX * kSizeY, 0.0f);
    std::vector<float> filter((2*HFS+1)*(2*HFS+1));

    // Populates input data structure by padded data
    for (size_t i = 0; i < kSizeY+2*HFS; ++i)
    {
        for (size_t j = 0; j < kSizeX+2*HFS; ++j)
        {
            input[i*(kSizeX+2*HFS) + j] = distribution(engine);
        }
    }

    // Creates the filter of random coeficients
    for (size_t i = 0; i < filter.size(); i++)
        filter[i] = distribution(engine);

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    //tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("Convolution", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("ConvolutionReference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Convolution", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("ConvolutionReference", referenceDefinition);

    std::vector<uint64_t> blockRange;
    std::vector<uint64_t> wptRange;

    // fake tuning parameters, encoding input
    tuner.AddParameter(kernel, "IMAGE_WIDTH", std::vector<uint64_t>{kSizeX});
    tuner.AddParameter(kernel, "IMAGE_HEIGHT", std::vector<uint64_t>{kSizeY});
    tuner.AddParameter(kernel, "HFS", std::vector<uint64_t>{HFS});

    // tuning parameters
    tuner.AddParameter(kernel, "BLOCK_SIZE_X", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128});
    tuner.AddParameter(kernel, "BLOCK_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "TILE_SIZE_X", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    tuner.AddParameter(kernel, "TILE_SIZE_Y", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    tuner.AddParameter(kernel, "PADDING", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "READ_ONLY", std::vector<uint64_t>{0, 1});

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector)
    {
        return (((size+vector.at(0)-1) / vector.at(0))+vector.at(1)-1) / vector.at(1);
    };

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK_SIZE_X", "TILE_SIZE_X"},
        globalModifier);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"BLOCK_SIZE_Y", "TILE_SIZE_Y"},
        globalModifier);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "BLOCK_SIZE_Y", ktt::ModifierAction::Multiply);

    // Add constraints
    auto minWGConstraint = [](const std::vector<uint64_t>& v) {return v[0] * v[1] >= 64;};
    tuner.AddConstraint(kernel, {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}, minWGConstraint);
    auto maxWGConstraint = [](const std::vector<uint64_t>& v) {return v[0] * v[1] <= 1024;};
    tuner.AddConstraint(kernel, {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}, maxWGConstraint);
    auto tileConstraint = [](const std::vector<uint64_t>& v) {return v[0] *v[1] < 30;};
    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "TILE_SIZE_Y"}, tileConstraint);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId filterId = tuner.AddArgumentVector(filter, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId outputId = tuner.AddArgumentVector(output, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId sizeXId = tuner.AddArgumentScalar(kSizeX);
    const ktt::ArgumentId sizeYId = tuner.AddArgumentScalar(kSizeY);
    const ktt::ArgumentId filterXId = tuner.AddArgumentScalar(HFS*2+1);
    const ktt::ArgumentId filterYId = tuner.AddArgumentScalar(HFS*2+1);

    tuner.SetArguments(definition, {outputId, inputId, filterId}); 
    tuner.SetArguments(referenceDefinition, {outputId, inputId, filterId, sizeXId, sizeYId, filterXId, filterYId}); 

    if constexpr (!rapidTest)
    {
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
        tuner.SetReferenceKernel(outputId, referenceKernel, ktt::KernelConfiguration());
    }

    // Launch kernel tuning
    const auto results = tuner.Tune(kernel);
    tuner.SaveResults(results, "KTConvolutionOutput", ktt::OutputFormat::XML);
    tuner.SaveResults(results, "KTConvolutionOutput", ktt::OutputFormat::JSON);

    return 0;
};
