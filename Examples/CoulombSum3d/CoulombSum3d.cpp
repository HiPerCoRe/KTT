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
    const std::string defaultKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3d.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3dReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
    const std::string defaultMlModel = kernelPrefix + "../Examples/CoulombSum3d/Models/2080-coulomb_output_DT.sav";
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3d.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3dReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = true;

// Add denser values to tuning parameters (useDenseParameters = true).
const bool useDenseParameters = false;

// Add wider ranges of tuning parameters (useWideParameters  = true).
const bool useWideParameters = false;

// Toggle usage of profile-based searcher
const bool useProfileSearcher = false;

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

    // Declare and initialize data
    const int gridSize = 256;
    int atoms = 64;

    const ktt::DimensionVector referenceNdRangeDimensions(gridSize / 16, gridSize / 16, gridSize);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);
    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions;

    std::vector<float> atomInfoX(atoms);
    std::vector<float> atomInfoY(atoms);
    std::vector<float> atomInfoZ(atoms);
    std::vector<float> atomInfoW(atoms);
    std::vector<float> atomInfo(4 * atoms);
    std::vector<float> energyGrid(gridSize * gridSize * gridSize, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 20.0f);
    const float gridSpacing = 0.5f; // in Angstroms

    for (int i = 0; i < atoms; ++i)
    {
        atomInfoX[i] = distribution(engine);
        atomInfoY[i] = distribution(engine);
        atomInfoZ[i] = distribution(engine);
        atomInfoW[i] = distribution(engine) / 40.0f;
        atomInfo[4 * i] = atomInfoX[i];
        atomInfo[4 * i + 1] = atomInfoY[i];
        atomInfo[4 * i + 2] = atomInfoZ[i];
        atomInfo[4 * i + 3] = atomInfoW[i];
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    }
    else
    {
        tuner.SetCompilerOptions("-use_fast_math");

        if constexpr (useProfiling)
        {
            printf("Executing with profiling switched ON.\n");
            tuner.SetProfiling(true);
        }
    }

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, ndRangeDimensions, workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("directCoulombSumReference", referenceKernelFile,
        referenceNdRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("CoulombSumReference", referenceDefinition);

    const ktt::ArgumentId aiId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aixId = tuner.AddArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiyId = tuner.AddArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aizId = tuner.AddArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiwId = tuner.AddArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aId = tuner.AddArgumentScalar(atoms);
    const ktt::ArgumentId gsId = tuner.AddArgumentScalar(gridSpacing);
    const ktt::ArgumentId gridDim = tuner.AddArgumentScalar(gridSize);
    const ktt::ArgumentId gridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);

    if constexpr (!useDenseParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{16, 32});
    }
    else
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{8, 16, 24, 32});
    }

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::DivideCeil);

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8});
        //tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{2,4});
    }
    else if constexpr (!useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    }
    else
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    }

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::DivideCeil);
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Z", std::vector<uint64_t>{1});

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
        //tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{1, 2});
    }
    else
    {
        tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    }

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS",
        ktt::ModifierAction::DivideCeil);

    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        //tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0});
    }
    else
    {
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    }

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1, 2 , 4, 8, 16});

        auto vec = [](const std::vector<uint64_t>& vector) {return vector.at(0) || vector.at(1) == 1; };
        tuner.AddConstraint(kernel, { "USE_SOA", "VECTOR_SIZE" }, vec);
    }
    else
    {
        // Not implemented in CUDA
        tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
        //tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1});
    }

    auto lt = [](const std::vector<uint64_t>& vector) {return vector.at(0) < vector.at(1);};
    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, lt);

    auto par = [](const std::vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
    tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);

    tuner.SetArguments(definition, std::vector<ktt::ArgumentId>{aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridDim, gridId});
    tuner.SetArguments(referenceDefinition, std::vector<ktt::ArgumentId>{aiId, aId, gsId, gridDim, gridId});

    if constexpr (!useProfiling && !rapidTest)
    {
        //TODO: this is temporary hack, there should be composition of zeroizing and Coulomb kernel,
        // otherwise, multiple profiling runs corrupt results
        tuner.SetReferenceKernel(gridId, referenceKernel, ktt::KernelConfiguration());
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);
    }
    tuner.SetSearcher(kernel, std::make_unique<ktt::DeterministicSearcher>());

#if KTT_CUDA_EXAMPLE
    if constexpr (useProfileSearcher)
    {
        tuner.SetProfileBasedSearcher(kernel, defaultMlModel, false);
    }
#endif

    const auto results = tuner.Tune(kernel/*, std::make_unique<ktt::ConfigurationCount>(1)*/);
    tuner.SaveResults(results, "CoulombSumOutput", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "CoulombSumOutput", ktt::OutputFormat::XML);

    return 0;
}
