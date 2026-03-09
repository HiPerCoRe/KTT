/******************************************************************************
 * This is a dummy example -- it uses fairy efficient version of 3D Coulomb sum
 * but does not tune anything.
 * The reason of existence of this example is testing stability of time and
 * power measurement.
 */

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <stdlib.h>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Dummy/Dummy.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Dummy/Dummy.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Sleep in the manipulator (can be randomized to 0, sleepDuration)
// (makes power measurement more challenging due to changes in GPU temperature)
const unsigned int sleepDuration = 1000;
const bool randomizeSleep = true;

// Toggle kernel profiling.
const bool useProfiling = false;

// Toggle robust power measurement (requires KTT built with --power-usage option).
const bool useRobustPowerMeasurement = true;

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
    //int atoms = 64;
    int atoms = 1024;

    const ktt::DimensionVector ndRangeDimensions(gridSize / 32, gridSize / 4, gridSize);
    const ktt::DimensionVector workGroupDimensions(32, 4);

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

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);

    tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
    {
        /*uint64_t sleep = sleepDuration;
        if (randomizeSleep)
            sleep = (sleep*rand())/RAND_MAX;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep));*/
        interface.RunKernel(definition);
    });

    const ktt::ArgumentId aiId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aixId = tuner.AddArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiyId = tuner.AddArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aizId = tuner.AddArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aiwId = tuner.AddArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId aId = tuner.AddArgumentScalar(atoms);
    const ktt::ArgumentId gsId = tuner.AddArgumentScalar(gridSpacing);
    const ktt::ArgumentId gridDim = tuner.AddArgumentScalar(gridSize);
    const ktt::ArgumentId gridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);

    // Create a tuning space big enough to not be exhausted until TuningDuration
    tuner.AddParameter(kernel, "DUMMY_1", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    tuner.AddParameter(kernel, "DUMMY_2", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    tuner.AddParameter(kernel, "DUMMY_3", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    tuner.AddParameter(kernel, "DUMMY_4", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    tuner.SetArguments(definition, std::vector<ktt::ArgumentId>{aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridDim, gridId});

    tuner.SetSearcher(kernel, std::make_unique<ktt::DeterministicSearcher>());

    // Configure robust power measurement parameters if enabled
    std::optional<ktt::PowerMeasurementParameters> powerParams;
    if constexpr (useRobustPowerMeasurement)
    {
        // Minimum 2000ms, maximum 20000ms, 0.5% tolerance
        powerParams = ktt::PowerMeasurementParameters(2000, 20000, 0.005);
    }

    const auto results = tuner.Tune(kernel, std::make_unique<ktt::TuningDuration>(600), powerParams);
    tuner.SaveResults(results, "DummyOutput", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "DummyOutput", ktt::OutputFormat::XML);

    return 0;
}
