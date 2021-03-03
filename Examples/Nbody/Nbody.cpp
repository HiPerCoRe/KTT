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
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Nbody/Nbody.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Nbody/NbodyReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Nbody/Nbody.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Nbody/NbodyReference.cl";
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
    int numberOfBodies = 128 * 1024;

    if constexpr (useProfiling)
    {
        numberOfBodies /= 8;
    }

    // Total NDRange size matches number of grid points
    const ktt::DimensionVector ndRangeDimensions(numberOfBodies);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(64);

    const float timeDelta = 0.001f;
    const float damping = 0.5f;
    const float softeningSqr = 0.1f * 0.1f;
    std::vector<float> oldBodyInfo(4 * numberOfBodies);
    std::vector<float> oldPosX(numberOfBodies);
    std::vector<float> oldPosY(numberOfBodies);
    std::vector<float> oldPosZ(numberOfBodies);
    std::vector<float> bodyMass(numberOfBodies);

    std::vector<float> newBodyInfo(4 * numberOfBodies, 0.f);

    std::vector<float> oldBodyVel(4 * numberOfBodies);
    std::vector<float> newBodyVel(4 * numberOfBodies);
    std::vector<float> oldVelX(numberOfBodies);
    std::vector<float> oldVelY(numberOfBodies);
    std::vector<float> oldVelZ(numberOfBodies);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 20.0f);

    for (int i = 0; i < numberOfBodies; ++i)
    {
        oldPosX[i] = distribution(engine);
        oldPosY[i] = distribution(engine);
        oldPosZ[i] = distribution(engine);
        bodyMass[i] = distribution(engine);

        oldVelX[i] = distribution(engine);
        oldVelY[i] = distribution(engine);
        oldVelZ[i] = distribution(engine);

        oldBodyInfo[4 * i] = oldPosX[i];
        oldBodyInfo[4 * i + 1] = oldPosY[i];
        oldBodyInfo[4 * i + 2] = oldPosZ[i];
        oldBodyInfo[4 * i + 3] = bodyMass[i];

        oldBodyVel[4 * i] = oldVelX[i];
        oldBodyVel[4 * i + 1] = oldVelY[i];
        oldBodyVel[4 * i + 2] = oldVelZ[i];
        oldBodyVel[4 * i + 3] = 0.0f;
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    }
    else
    {
        tuner.SetCompilerOptions("-use_fast_math");
    }

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("nbody_kernel", kernelFile, ndRangeDimensions, workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("nbody_kernel_reference", referenceKernelFile, ndRangeDimensions,
        referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Nbody", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("NbodyReference", referenceDefinition);

    // Multiply work-group size in dimensions x and y by two parameters that follow (effectively setting work-group size to parameters' values)
    if constexpr (!useDenseParameters && !useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{64, 128, 256, 512});
    }
    else if constexpr (!useWideParameters)
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{64, 80, 96, 112, 128,160, 192, 224, 256, 320, 384, 448, 512});
    }
    else
    {
        tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{32, 64, 80, 96, 112, 128,160, 192, 224, 256, 320, 384, 448,
            512, 640, 768, 894, 1024});
    }

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);

    if constexpr (!useWideParameters)
    {
        tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8});
    }
    else
    {
        tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    }

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
        ktt::ModifierAction::Divide);

    if constexpr (!useDenseParameters)
    {
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR1", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR2", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
    }
    else
    {
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR1", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
        tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR2", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});
    }

    tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});

    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8, 16});
    }
    else
    {
        tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
    }

    // Add all arguments utilized by kernels
    const ktt::ArgumentId oldBodyInfoId = tuner.AddArgumentVector(oldBodyInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldPosXId = tuner.AddArgumentVector(oldPosX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldPosYId = tuner.AddArgumentVector(oldPosY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldPosZId = tuner.AddArgumentVector(oldPosZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId massId = tuner.AddArgumentVector(bodyMass, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId newBodyInfoId = tuner.AddArgumentVector(newBodyInfo, ktt::ArgumentAccessType::WriteOnly);

    const ktt::ArgumentId oldVelId = tuner.AddArgumentVector(oldBodyVel, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldVelXId = tuner.AddArgumentVector(oldVelX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldVelYId = tuner.AddArgumentVector(oldVelY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId oldVelZId = tuner.AddArgumentVector(oldVelZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId newBodyVelId = tuner.AddArgumentVector(newBodyVel, ktt::ArgumentAccessType::WriteOnly);

    const ktt::ArgumentId deltaTimeId = tuner.AddArgumentScalar(timeDelta);
    const ktt::ArgumentId dampingId = tuner.AddArgumentScalar(damping);
    const ktt::ArgumentId softeningSqrId = tuner.AddArgumentScalar(softeningSqr);
    const ktt::ArgumentId numberOfBodiesId = tuner.AddArgumentScalar(numberOfBodies);

    // Add conditions
    auto lteq = [](const std::vector<uint64_t>& vector) {return vector.at(0) <= vector.at(1);};
    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, lteq);
    auto lteq256 = [](const std::vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) <= 256;};
    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, lteq256);
    auto vectorizedSoA = [](const std::vector<uint64_t>& vector) {return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1);};
    tuner.AddConstraint(kernel, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.SetArguments(definition, std::vector<ktt::ArgumentId>{deltaTimeId,
        oldBodyInfoId, oldPosXId, oldPosYId, oldPosZId, massId, newBodyInfoId, // position
        oldVelId, oldVelXId, oldVelYId, oldVelZId, newBodyVelId, // velocity
        dampingId, softeningSqrId, numberOfBodiesId});
    tuner.SetArguments(referenceDefinition, std::vector<ktt::ArgumentId>{deltaTimeId, oldBodyInfoId, newBodyInfoId, oldVelId, newBodyVelId,
        dampingId, softeningSqrId});

    if constexpr (!rapidTest)
    {
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001);
        tuner.SetReferenceKernel(newBodyVelId, referenceKernel, ktt::KernelConfiguration());
        tuner.SetReferenceKernel(newBodyInfoId, referenceKernel, ktt::KernelConfiguration());
    }

    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "NbodyOutput", ktt::OutputFormat::JSON);

    return 0;
}
