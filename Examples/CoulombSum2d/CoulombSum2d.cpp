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

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels.
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Examples/CoulombSum2d/CoulombSum2d.cl";
    std::string referenceKernelFile = kernelPrefix + "../Examples/CoulombSum2d/CoulombSum2dReference.cl";

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

    // Declare kernel parameters.
    const ktt::DimensionVector ndRangeDimensions(512, 512);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);
    // Total NDRange size matches number of grid points.
    const size_t numberOfGridPoints = ndRangeDimensions.GetSizeX() * ndRangeDimensions.GetSizeY();
    // If higher than 4k, computations with constant memory enabled will be invalid on many devices due to constant memory capacity limit.
    const int numberOfAtoms = 4000;

    // Declare data variables.
    const float gridSpacing = 0.5f;
    std::vector<float> atomInfo(4 * numberOfAtoms);
    std::vector<float> atomInfoX(numberOfAtoms);
    std::vector<float> atomInfoY(numberOfAtoms);
    std::vector<float> atomInfoZ(numberOfAtoms);
    std::vector<float> atomInfoW(numberOfAtoms);
    std::vector<float> energyGrid(numberOfGridPoints, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 40.0f);

    for (size_t i = 0; i < static_cast<size_t>(numberOfAtoms); ++i)
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

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Add two kernels to tuner, one of the kernels acts as a reference kernel.
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);

    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("directCoulombSumReference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("CoulombSumReference", referenceDefinition);

    // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers.
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1, 2});

    // Using vectorized SoA only makes sense when vectors are longer than 1.
    auto vectorizedSoA = [](const std::vector<size_t>& vector) {return vector[0] > 1 || vector[1] != 2;}; 
    tuner.AddConstraint(kernel, {"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR.
    tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
        ktt::ModifierAction::Divide);

    // Multiply work-group size in dimensions x and y by the following parameters (effectively setting work-group size to their values).
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{4, 8, 16, 32});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);

    // Add all kernel arguments.
    ktt::ArgumentId atomInfoId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoXId = tuner.AddArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoYId = tuner.AddArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoZId = tuner.AddArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoWId = tuner.AddArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId numberOfAtomsId = tuner.AddArgumentScalar(numberOfAtoms);
    ktt::ArgumentId gridSpacingId = tuner.AddArgumentScalar(gridSpacing);
    ktt::ArgumentId energyGridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set arguments for both tuned and reference kernel definitions, order of arguments is important.
    tuner.SetArguments(definition, {atomInfoId, atomInfoXId, atomInfoYId, atomInfoZId, atomInfoWId, numberOfAtomsId, gridSpacingId,
        energyGridId});
    tuner.SetArguments(referenceDefinition, {atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId});

    // Set searcher to random.
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());

    // Specify custom tolerance threshold for validation of floating-point arguments. Default threshold is 1e-4.
    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    // Set reference kernel which validates results provided by the tuned kernel.
    tuner.SetReferenceKernel(energyGridId, referenceKernel, ktt::KernelConfiguration());

    // Launch kernel tuning, end after 1 minute.
    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel, std::make_unique<ktt::TuningDuration>(60.0));

    // Save tuning results to JSON file.
    tuner.SaveResults(results, "CoulombSum2dOutput", ktt::OutputFormat::JSON);

    return 0;
}
