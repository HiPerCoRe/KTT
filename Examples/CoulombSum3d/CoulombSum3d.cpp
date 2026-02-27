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
    const auto computeApi = ktt::ComputeApi::CUDA;
    const std::string defaultMlModel = kernelPrefix + "../Examples/CoulombSum3d/Models/2080-coulomb_output_DT.sav";
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3d.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#elif KTT_CPP_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/CoulombSum3d/CoulombSum3d.cppkernel";
    const auto computeApi = ktt::ComputeApi::Cpp;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Toggle usage of profile-based searcher
const bool useProfileSearcher = false;

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;

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

    // Declare and initialize data
    const int gridSize = 256;
    int atoms = 256;//4096;

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
    //tuner.SetLoggingLevel(ktt::LoggingLevel::Debug);

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    }
    else if constexpr (computeApi == ktt::ComputeApi::CUDA)
    {
        tuner.SetCompilerOptions("-use_fast_math");

        if constexpr (useProfiling)
        {
            printf("Executing with profiling switched ON.\n");
            tuner.SetProfiling(true);
        }
    }
    else
    {
        tuner.SetCompilerOptions("-O3 -ffast-math -march=native -fopenmp");
    }

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, ndRangeDimensions, workGroupDimensions);
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);

    const ktt::ArgumentId aiId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aixId = tuner.AddArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiyId = tuner.AddArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aizId = tuner.AddArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiwId = tuner.AddArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aId = tuner.AddArgumentScalar(atoms);
    const ktt::ArgumentId gsId = tuner.AddArgumentScalar(gridSpacing);
    const ktt::ArgumentId gridDim = tuner.AddArgumentScalar(gridSize);
    const ktt::ArgumentId gridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);

    if constexpr (computeApi == ktt::ComputeApi::OpenCL || computeApi == ktt::ComputeApi::CUDA)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{16, 32});
        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);
        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::DivideCeil);
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8});
        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);
        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::DivideCeil);
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Z", std::vector<uint64_t>{1});
        tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS",
            ktt::ModifierAction::DivideCeil);    
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});

        auto lt = [](const std::vector<uint64_t>& vector) {return vector.at(0) < vector.at(1);};
        tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, lt);
        auto par = [](const std::vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
        tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);

        if constexpr (computeApi == ktt::ComputeApi::OpenCL)
        {
            tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0, 1});
            tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
            tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1, 2 , 4, 8, 16});

            auto vec = [](const std::vector<uint64_t>& vector) {return vector.at(0) || vector.at(1) == 1; };
            tuner.AddConstraint(kernel, { "USE_SOA", "VECTOR_SIZE" }, vec);
        }
        else if constexpr (computeApi == ktt::ComputeApi::CUDA)
        {
            tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
            tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
            tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1});
        }
    }
    else // CPP
    {
        tuner.AddParameter(kernel, "OMP_COLLAPSE", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "OMP_SCHEDULING", std::vector<uint64_t>{0, 1, 2});
        tuner.AddParameter(kernel, "OMP_SCHED_CHUNK", std::vector<uint64_t>{2, 4, 8, 16, 32, 64, 128});
        tuner.AddParameter(kernel, "TILE_SIZE", std::vector<uint64_t>{0, 8, 16, 32, 64});

    }

    tuner.SetArguments(definition, std::vector<ktt::ArgumentId>{aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridDim, gridId});

    if constexpr (!useProfiling && !rapidTest)
    {
        //TODO: this is temporary hack, there should be composition of zeroizing and Coulomb kernel,
        // otherwise, multiple profiling runs corrupt results
        tuner.SetReferenceComputation(gridId, [&atomInfoW, atomInfoX, &atomInfoY, &atomInfoZ, atoms, gridSpacing, gridSize](void* buffer){
            float* grid = static_cast<float*>(buffer);
            #pragma omp parallel for
            for (int z = 0; z < gridSize; z++)
                for (int y = 0; y < gridSize; y++)
                    for (int x = 0; x < gridSize; x++) 
                    {
                        float e = 0.0f;
                        float gx = (float)x * gridSpacing;
                        float gy = (float)y * gridSpacing;
                        float gz = (float)z * gridSpacing;
                        for (int a = 0; a < atoms; a++)
                            e += atomInfoW[a] * (1.0f/sqrtf((atomInfoX[a]-gx)*(atomInfoX[a]-gx) + (atomInfoY[a]-gy)*(atomInfoY[a]-gy) 
                                + (atomInfoZ[a]-gz)*(atomInfoZ[a]-gz)));
                        grid[z*gridSize*gridSize + y*gridSize + x] = e;
                    }
        });
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);
    }
    tuner.SetSearcher(kernel, std::make_unique<ktt::DeterministicSearcher>());

#if KTT_CUDA_EXAMPLE
    if constexpr (useProfileSearcher)
    {
        tuner.SetProfileBasedSearcher(kernel, defaultMlModel, false);
    }
#endif

    const auto results = tuner.Tune(kernel, std::make_unique<ktt::FailureFraction>(0.1, 10) /*std::make_unique<ktt::ConfigurationCount>(2)*/);
    tuner.SaveResults(results, "CoulombSumOutput", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "CoulombSumOutput_T4", ktt::OutputFormat::JSON_T4);
    tuner.SaveResults(results, "CoulombSumOutput", ktt::OutputFormat::XML);

    double bestDuration = std::numeric_limits<double>::max();
    std::string bestConfig;
    for (const auto& result : results)
    {
        double duration = result.GetTotalDuration();
        std::string config = result.GetConfiguration().GetString();
        
        std::cout << "Configuration: " << config
                  << " -> Duration: " << duration << " ns"
                  << " -> Status: " << (result.GetStatus() == ktt::ResultStatus::Ok ? "OK" : "FAILED")
                  << std::endl;
        
        if (result.GetStatus() == ktt::ResultStatus::Ok && duration < bestDuration)
        {
            bestDuration = duration;
            bestConfig = config;
        }
    }

    if (!results.empty())
    {
        std::cout << "\nBest configuration: " << bestConfig << " with duration " << bestDuration << " ns (" << 1000.0*(double)atoms*(double)gridSize*(double)gridSize*(double)gridSize/bestDuration << "mevals/s)" << std::endl;
    }

    return 0;
}
