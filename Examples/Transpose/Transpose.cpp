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
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Transpose/Transpose.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Transpose/TransposeReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Transpose/Transpose.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Transpose/TransposeReference.cl";
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

int main(int argc, char **argv)
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

    // Declare and initialize data
    int width;
    int height;

    if constexpr (!useProfiling)
    {
        width = 8192;
        height = 8192;
    }
    else
    {
        width = 4096;
        height = 4096;
    }

    const ktt::DimensionVector ndRangeDimensions(width, height);
    const ktt::DimensionVector ndRangeDimensionsReference(width / 16, height / 16);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare data variables
    std::vector<float> dst(width * height);
    std::vector<float> src(width * height);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

    for (int i = 0; i < width * height; ++i)
    {
        src[i] = distribution(engine);
    }

    // Create tuner
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Create kernel and configure input/output
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("mtran", kernelFile, ndRangeDimensions,
        ktt::DimensionVector(1, 1));
    const ktt::KernelId referenceDefinition = tuner.AddKernelDefinitionFromFile("mtranReference", referenceKernelFile,
        ndRangeDimensionsReference, referenceWorkGroupDimensions);
    
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Transposition", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("TranspositionReference", referenceDefinition);

    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId dstId = tuner.AddArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId widthId = tuner.AddArgumentScalar(width);
    const ktt::ArgumentId heightId = tuner.AddArgumentScalar(height);
    
    tuner.SetArguments(definition, {dstId, srcId, widthId, heightId});
    tuner.SetArguments(referenceDefinition, {dstId, srcId, widthId, heightId});

    // Create tuning space
    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8});
        tuner.AddParameter(kernel, "PREFETCH", std::vector<uint64_t>{0, 1, 2});
    }
    else
    {
        tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
    }

    tuner.AddParameter(kernel, "CR", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "PADD_LOCAL", std::vector<uint64_t>{0, 1});

    std::vector<uint64_t> sizeRanges;

    if constexpr (!useWideParameters && !useDenseParameters)
    {
        sizeRanges = {1, 2, 4, 8, 16, 32, 64};
    }
    else if constexpr (!useWideParameters)
    {
        sizeRanges = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
    }
    else
    {
        sizeRanges = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128};
    }

    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", sizeRanges);
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", sizeRanges);
    tuner.AddParameter(kernel, "TILE_SIZE_X", sizeRanges);
    tuner.AddParameter(kernel, "TILE_SIZE_Y", sizeRanges);
    tuner.AddParameter(kernel, "DIAGONAL_MAP", std::vector<uint64_t>{0, 1});
    
    // Constraint tuning space
    auto xConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] == v[1]); };
    auto yConstraint = [] (const std::vector<uint64_t>& v) { return (v[1] <= v[0]); };
    auto tConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
    auto pConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] || !v[1]); };
    auto vlConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || v[1] == 1); };

    uint64_t maxMult = 64;

    if constexpr (useWideParameters)
    {
        maxMult = 128;
    }

    auto vConstraint = [maxMult] (const std::vector<uint64_t>& v) { return (v[0]*v[1] <= maxMult); };
    
    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "WORK_GROUP_SIZE_X"}, xConstraint);
    tuner.AddConstraint(kernel, {"TILE_SIZE_Y", "WORK_GROUP_SIZE_Y"}, yConstraint);
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, tConstraint);
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "PADD_LOCAL"}, pConstraint);
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "VECTOR_TYPE"}, vlConstraint);
    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "VECTOR_TYPE"}, vConstraint);

    // Configure parallelism
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);
    auto xGlobalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector) {return size / vector.at(0) / vector.at(1);};
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"TILE_SIZE_X", "VECTOR_TYPE"}, xGlobalModifier);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y",
        ktt::ModifierAction::Divide);

    auto wgSize = [](const std::vector<uint64_t>& v) {return v[0]*v[1] >= 32;};
    tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, wgSize);

    if constexpr (!rapidTest)
    {
        tuner.SetReferenceKernel(dstId, referenceKernel, ktt::KernelConfiguration());
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
    }

    // Perform tuning
    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "TranspositionOutput.csv", ktt::OutputFormat::JSON);

    return 0;
}
