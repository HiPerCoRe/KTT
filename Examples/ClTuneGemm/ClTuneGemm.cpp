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
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemm.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemmReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemm.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/ClTuneGemm/ClTuneGemmReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Reduced tuning parameters set, taken from CLTune.
const bool useReducedSet = false;

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return ((a / b) * b == a) ? true : false;
};

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

    uint32_t kSizeM;
    uint32_t kSizeN;
    uint32_t kSizeK;

    if constexpr (!useProfiling)
    {
        kSizeM = 2048;
        kSizeN = 2048;
        kSizeK = 2048;
    }
    else
    {
        kSizeM = 2048 / 2;
        kSizeN = 2048 / 2;
        kSizeK = 2048 / 2;
    }

    const ktt::DimensionVector ndRangeDimensions(kSizeM, kSizeN);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    std::vector<float> mat_a(kSizeM * kSizeK);
    std::vector<float> mat_b(kSizeN * kSizeK);
    std::vector<float> mat_c(kSizeM * kSizeN);

    for (uint32_t i = 0; i < kSizeM * kSizeK; ++i)
    {
        mat_a[i] = distribution(engine);
    }
        
    for (uint32_t i = 0; i < kSizeN * kSizeK; ++i)
    {
        mat_b[i] = distribution(engine);
    }

    for (uint32_t i = 0; i < kSizeM * kSizeN; ++i)
    {
        mat_c[i] = 0.0f;
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
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("gemm_fast", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("gemm_reference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Gemm", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("GemmReference", referenceDefinition);

    if constexpr (useReducedSet)
    {
        tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64});
        tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64});
        tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{32});
        tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4});
        tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4});
        tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});
    }
    else
    {
        tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64, 128});
        tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64, 128});
        tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{16, 32});
        tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
        tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2, 8});

        if constexpr (computeApi == ktt::ComputeApi::OpenCL)
        {
            tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4, 8});
            tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4, 8});
        }
        else
        {
            tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4});
            tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4});
        }

        tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});
    }

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector) {return size * vector.at(0) / vector.at(1);};
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"MDIMC", "MWG"}, globalModifier);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"NDIMC", "NWG"}, globalModifier);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "MDIMC", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "NDIMC", ktt::ModifierAction::Multiply);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId kSizeMId = tuner.AddArgumentScalar(kSizeM);
    const ktt::ArgumentId kSizeNId = tuner.AddArgumentScalar(kSizeN);
    const ktt::ArgumentId kSizeKId = tuner.AddArgumentScalar(kSizeK);
    const ktt::ArgumentId matAId = tuner.AddArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId matBId = tuner.AddArgumentVector(mat_b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId matCId = tuner.AddArgumentVector(mat_c, ktt::ArgumentAccessType::WriteOnly);

    // Add conditions
    // Sets constraints: Set-up the constraints functions to use. The constraints require a function
    // object (in this case a lambda) which takes a vector of tuning parameter values and returns
    // a boolean value whether or not the tuning configuration is legal. In this case, the helper
    // function 'IsMultiple' is employed for convenience. In the calls to 'AddConstraint' below, the
    // vector of parameter names (as strings) matches the input integer vector of the lambda's.
    auto multipleOfX = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1]);};
    auto multipleOfXMulY = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1] * v[2]);};
    auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], (v[1] * v[2]) / v[3]);};

    // Sets constraints: Requirement for unrolling the KWG loop
    tuner.AddConstraint(kernel, {"KWG", "KWI"}, multipleOfX);

    // Sets constraints: Required for integer MWI and NWI
    tuner.AddConstraint(kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);

    // Sets constraints: Required for integer MWIA and NWIB
    tuner.AddConstraint(kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);

    // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);

    tuner.SetArguments(definition, {kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId});
    tuner.SetArguments(referenceDefinition, {kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId});

    if constexpr (!rapidTest)
    {
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001);
        tuner.SetReferenceKernel(matCId, referenceKernel, ktt::KernelConfiguration());
    }

    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "GemmOutput", ktt::OutputFormat::XML);

    return 0;
};
