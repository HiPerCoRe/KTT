#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    const std::string kernelFilePrefix = "";
    const std::string profileSearcherDir = "../profile-searcher/";
#else
    const std::string kernelFilePrefix = "../";
    const std::string profileSearcherDir = "../../profile-searcher/";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cu";
    const std::string defaultReferenceKernelFile = kernelFilePrefix + "../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cu";
    const auto computeAPI = ktt::ComputeAPI::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cl";
    const std::string defaultReferenceKernelFile = kernelFilePrefix + "../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cl";
    const auto computeAPI = ktt::ComputeAPI::OpenCL;
#endif

// Rapid test switch off comparison with CPU implementation
#define RAPID_TEST 1

// Gathering hardware performance counters
#define USE_PROFILING 0

// Those macros enlarge tuning space by adding denser values to tuning 
// parameters (USE_DENSE_TUNPAR == 1), and also adding wider ranges of tuning
// parameters (USE_WIDE_TUNPAR  == 1)
#define USE_DENSE_TUNPAR 0
#define USE_WIDE_TUNPAR 0

// Switch on/off exhaustive search (complete exploration of the tuning space)
#define EXHAUSTIVE_SEARCH 0
#if EXHAUSTIVE_SEARCH == 0
    //XXX profile-searcher works with CUDA only
    #define USE_PROFILE_SEARCHER 1
#endif

#define TUNE_SEC 30

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
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

    // Declare kernel parameters
    const int gridSize = 256;
    const int atoms = /*4096*/256;
    const ktt::DimensionVector referenceNdRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);
    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions(1, 1);

    // Declare data variables
    float gridSpacing;
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
    gridSpacing = 0.5f; // in Angstroms

    for (int i = 0; i < atoms; i++)
    {
        atomInfoX.at(i) = distribution(engine);
        atomInfoY.at(i) = distribution(engine);
        atomInfoZ.at(i) = distribution(engine);
        atomInfoW.at(i) = distribution(engine) / 40.0f;
        atomInfo.at(i*4) = atomInfoX.at(i);
        atomInfo.at(i*4 + 1) = atomInfoY.at(i);
        atomInfo.at(i*4 + 2) = atomInfoZ.at(i);
        atomInfo.at(i*4 + 3) = atomInfoW.at(i);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeAPI);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    if (computeAPI == ktt::ComputeAPI::OpenCL)
    {
        tuner.setCompilerOptions("-cl-fast-relaxed-math");
    }
    else
    {
        tuner.setCompilerOptions("-use_fast_math");
        #if USE_PROFILING == 1
        printf("Executing with profiling switched ON.\n");
        tuner.setKernelProfiling(true);
        #endif
    }

    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "directCoulombSum", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "directCoulombSumReference", referenceNdRangeDimensions,
        referenceWorkGroupDimensions);

    ktt::ArgumentId aiId = tuner.addArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aixId = tuner.addArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aiyId = tuner.addArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aizId = tuner.addArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aiwId = tuner.addArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aId = tuner.addArgumentScalar(atoms);
    ktt::ArgumentId gsId = tuner.addArgumentScalar(gridSpacing);
    ktt::ArgumentId gridDim = tuner.addArgumentScalar(gridSize);
    ktt::ArgumentId gridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);

#if RAPID_TEST == 1
    tuner.persistArgument(aiId, true);
    tuner.persistArgument(aixId, true);
    tuner.persistArgument(aiyId, true);
    tuner.persistArgument(aizId, true);
    tuner.persistArgument(aiwId, true);
    tuner.persistArgument(gridDim, true);
    tuner.persistArgument(gridId, true);
#endif

    #if USE_DENSE_TUNPAR == 0
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {16, 32});
    #else
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {8, 16, 24, 32});
    #endif
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    #if USE_DENSE_TUNPAR == 0 && USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {1, 2, 4, 8});
    #else
        #if USE_WIDE_TUNPAR == 0
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {1, 2, 3, 4, 5, 6, 7, 8});
        #else
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
        #endif
    #endif
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Z", {1});
    #if USE_DENSE_TUNPAR == 0 && USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "Z_ITERATIONS", {1, 2, 4, 8, 16, 32});
    #else
        tuner.addParameter(kernelId, "Z_ITERATIONS", {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    #endif
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS", ktt::ModifierAction::DivideCeil);
    #if USE_DENSE_TUNPAR == 0 && USE_WIDE_TUNPAR == 0
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0, 1, 2, 4, 8, 16, 32});
    #else
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    #endif
    if (computeAPI == ktt::ComputeAPI::OpenCL)
    {
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0, 1});
        tuner.addParameter(kernelId, "USE_SOA", {0, 1});
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8, 16});
    }
    else
    {
        // not implemented in CUDA
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0});
        tuner.addParameter(kernelId, "USE_SOA", {0, 1});
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1});
    }

    auto lt = [](const std::vector<size_t>& vector) {return vector.at(0) <= vector.at(1);};
    tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, lt);
    auto vec = [](const std::vector<size_t>& vector) {return vector.at(0) || vector.at(1) == 1;};
    tuner.addConstraint(kernelId, {"USE_SOA", "VECTOR_SIZE"}, vec);
    auto par = [](const std::vector<size_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
    tuner.addConstraint(kernelId, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);

    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridDim, gridId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{aiId, aId, gsId, gridDim, gridId});

#if USE_PROFILING == 0 && RAPID_TEST == 0
    //TODO: this is temporal hack, there should be composition of zeroizing and coulomb kernel, otherwise, multiple profiling runs corrupt results
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{gridId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);
#endif

#if EXHAUSTIVE_SEARCH == 0
#if USE_PROFILE_SEARCHER == 1 and KTT_CUDA_EXAMPLE
    unsigned int ccMajor = tuner.getCurrentDeviceInfo().getCUDAComputeCapabilityMajor();
    unsigned int ccMinor = tuner.getCurrentDeviceInfo().getCUDAComputeCapabilityMinor();
    unsigned int myMP = tuner.getCurrentDeviceInfo().getMaxComputeUnits();
    auto searcher = std::make_unique<ktt::ProfileSearcher>(ccMajor*10 + ccMinor, myMP, "1070-coulomb", 61, ktt::ProfileSearcherModel::DecisionTree, profileSearcherDir);
    auto searcherRaw = searcher.get();
    tuner.setSearcher(kernelId, std::move(searcher));
#else
    tuner.setSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
#endif
#endif

    // Launch kernel tuning
#if EXHAUSTIVE_SEARCH == 1
    tuner.tuneKernel(kernelId); //XXX tuneKernel does not work with current implementation of profile-based searcher
#else
    //XXX in current implementation of profile-based searcher, the iterative profiling has to be performed
    std::vector<float> oneElement(1);
    ktt::OutputDescriptor output(gridId, (void*)oneElement.data(), 1*sizeof(float));
    int confTested = 0;
    int kernTested = 0;

    // loop for desired amount of time
    clock_t start = time(NULL);
    while (time(NULL) - start < TUNE_SEC) {
        // turn on/off profiling and gather statistics
#if USE_PROFILE_SEARCHER == 1 and KTT_CUDA_EXAMPLE
        if (searcherRaw->shouldProfile()) {
            tuner.setKernelProfiling(true);
            kernTested++;
        }
        else {
            tuner.setKernelProfiling(false);
            confTested++;
            kernTested++;
        }
#else
        confTested++;
        kernTested++;
#endif
        // tune kernel
        tuner.tuneKernelByStep(kernelId, {output});

        // dump time and best kernel
        ktt::ComputationResult bestConf = tuner.getBestComputationResult(kernelId);
        std::cout << "Execution after " << time(NULL) - start << " second(s), tested " << confTested << " configurations, best kernel " << bestConf.getDuration() << " ns" << std::endl;
    }
    std::cout << "Number of configurations tested: " << confTested << ", required kernel tests: " << kernTested << std::endl;
#endif

    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "coulomb_sum_3d_output.csv", ktt::PrintFormat::CSV);

    return 0;
}

