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
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneConvolution/ClTuneConvolution.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneConvolution/ClTuneConvolutionReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneConvolution/ClTuneConvolution.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneConvolution/ClTuneConvolutionReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Add denser values to tuning parameters (useDenseParameters = true).
const bool useDenseParameters = false;

// Add wider ranges of tuning parameters (useWideParameters  = true).
const bool useWideParameters = false;

// Settings (synchronise these with kernel source files)
const int HFS = 3; // Half filter size
const int FS = (HFS + HFS + 1); // Filter size

// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(const size_t a, const size_t b)
{
    return (a + b - 1) / b;
}

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
        kSizeX = 2048;
        kSizeY = 2048;
    }

    const ktt::DimensionVector ndRangeDimensions(kSizeX, kSizeY);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    std::vector<float> mat_a((kSizeX + 2 * HFS) * (kSizeY + 2 * HFS), 0.0f);
    std::vector<float> mat_b(kSizeX * kSizeY, 0.0f);
    std::vector<float> coeff(FS * FS);

    // Populates input data structure by padded data
    for (size_t i = 0; i < kSizeY; ++i)
    {
        for (size_t j = 0; j < kSizeX; ++j)
        {
            mat_a[(i + HFS) * (kSizeX + 2 * HFS) + j + HFS] = distribution(engine);
        }
    }

    // Creates the filter coefficients (gaussian blur)
    float sigma = 1.0f;
    float mean = FS / 2.0f;
    float sum = 0.0f;

    for (size_t x = 0; x < FS; ++x)
    {
        for (size_t y = 0; y < FS; ++y)
        {
            const float exponent = -0.5f * (pow((x - mean) / sigma, 2.0f) + pow((y - mean) / sigma, 2.0f));
            coeff[y * FS + x] = static_cast<float>(exp(exponent) / (2.0f * 3.14159265f * sigma * sigma));
            sum += coeff[y * FS + x];
        }
    }

    for (auto& item : coeff)
    {
        item = item / sum;
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("conv", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("conv_reference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Convolution", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("ConvolutionReference", referenceDefinition);

    std::vector<uint64_t> blockRange;
    std::vector<uint64_t> wptRange;

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        blockRange = {8, 16, 32, 64};
        wptRange = {1, 2, 4, 8};
    }
    else if constexpr (!useWideParameters)
    {
        blockRange = {8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
        wptRange = {1, 2, 3, 4, 5, 6, 7, 8};
    }
    else
    {
        blockRange = {8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
        wptRange = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};
    }

    tuner.AddParameter(kernel, "TBX", blockRange);
    tuner.AddParameter(kernel, "TBY", blockRange);
    tuner.AddParameter(kernel, "LOCAL", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "WPTX", wptRange);
    tuner.AddParameter(kernel, "WPTY", wptRange);
    tuner.AddParameter(kernel, "VECTOR", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, static_cast<uint64_t>(FS)});
    tuner.AddParameter(kernel, "PADDING", std::vector<uint64_t>{0, 1});

    // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
    // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
    // extra threads to compute the halo threads. How many extra threads are needed is dependend on
    // the filter size. Here we support a the TBX and TBY size plus up to 10 extra threads.
    std::vector<uint64_t> integers{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};

    tuner.AddParameter(kernel, "TBX_XL", integers);
    tuner.AddParameter(kernel, "TBY_XL", integers);

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector)
    {
        return (size * vector.at(0)) / (vector.at(1) * vector.at(2));
    };

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"TBX_XL", "TBX", "WPTX"},
        globalModifier);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"TBY_XL", "TBY", "WPTY"},
        globalModifier);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "TBX_XL", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "TBY_XL", ktt::ModifierAction::Multiply);

    // Add constraints
    auto haloThreads = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 2) {return (v[1] == v[2] + CeilDiv(2 * HFS, v[3]));} // With halo threads
        else {return (v[1] == v[2]);} // Without halo threads
    };

    tuner.AddConstraint(kernel, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, haloThreads);
    tuner.AddConstraint(kernel, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, haloThreads);

    // Sets the constrains on the vector size
    auto vectorConstraint = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 2) {return IsMultiple(v[2], v[1]) && IsMultiple(2 * HFS, v[1]);}
        else {return IsMultiple(v[2], v[1]);}
    };

    tuner.AddConstraint(kernel, {"LOCAL", "VECTOR", "WPTX"}, vectorConstraint);

    // Sets padding to zero in case local memory is not used
    auto paddingConstraint = [](const std::vector<uint64_t>& v) {return (v[1] == 0 || v[0] != 0);};
    tuner.AddConstraint(kernel, {"LOCAL", "PADDING"}, paddingConstraint);

    // Ensure divisibility
    auto divConstraint = [](const std::vector<uint64_t>& v) {return v[0] % v[1] == 0;};
    tuner.AddConstraint(kernel, {"TBX", "WPTX"}, divConstraint);
    tuner.AddConstraint(kernel, {"TBY", "WPTY"}, divConstraint);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId kSizeXId = tuner.AddArgumentScalar(kSizeX);
    const ktt::ArgumentId kSizeYId = tuner.AddArgumentScalar(kSizeY);
    const ktt::ArgumentId matAId = tuner.AddArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId coeffId = tuner.AddArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId matBId = tuner.AddArgumentVector(mat_b, ktt::ArgumentAccessType::WriteOnly);

    tuner.SetArguments(definition, {kSizeXId, kSizeYId, matAId, coeffId, matBId}); 
    tuner.SetArguments(referenceDefinition, {kSizeXId, kSizeYId, matAId, coeffId, matBId}); 

    if constexpr (!rapidTest)
    {
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
        tuner.SetReferenceKernel(matBId, referenceKernel, ktt::KernelConfiguration());
    }

    // Launch kernel tuning
    const auto results = tuner.Tune(kernel);
    tuner.SaveResults(results, "ConvolutionOutput", ktt::OutputFormat::XML);

    return 0;
};
