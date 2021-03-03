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
    std::string kernelFile = kernelPrefix + "../Examples/CoulombSum3dIterative/CoulombSum3dIterative.cl";
    std::string referenceKernelFile = kernelPrefix + "../Examples/CoulombSum3dIterative/CoulombSum3dIterativeReference.cl";

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

    // Set the problem size and declare data variables
    const int atoms = 4000;
    const int gridSize = 256;
    const float gridSpacing = 0.5f;
    const int zIndex = 0;

    std::vector<float> atomInfo(4 * atoms);
    std::vector<float> atomInfoPrecomp(4 * atoms);
    std::vector<float> atomInfoX(atoms);
    std::vector<float> atomInfoY(atoms);
    std::vector<float> atomInfoZ(atoms);
    std::vector<float> atomInfoZ2(atoms);
    std::vector<float> atomInfoW(atoms);
    std::vector<float> energyGrid(gridSize * gridSize * gridSize, 0.0f);

    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 40.0f);

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

        atomInfoPrecomp[4 * i] = atomInfoX[i];
        atomInfoPrecomp[4 * i + 1] = atomInfoY[i];
        // Do not store z, it will be rewritten anyway
        atomInfoPrecomp[4 * i + 3] = atomInfoW[i];
    }

    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    
    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("directCoulombSumReference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("CoulombSumReference", referenceDefinition);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId atomInfoId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoPrecompId = tuner.AddArgumentVector(atomInfoPrecomp, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoXId = tuner.AddArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoYId = tuner.AddArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoZId = tuner.AddArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoZ2Id = tuner.AddArgumentVector(atomInfoZ2, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomInfoWId = tuner.AddArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId numberOfAtomsId = tuner.AddArgumentScalar(atoms);
    const ktt::ArgumentId gridSpacingId = tuner.AddArgumentScalar(gridSpacing);
    const ktt::ArgumentId zIndexId = tuner.AddArgumentScalar(zIndex);
    const ktt::ArgumentId energyGridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.SetArguments(definition, std::vector<ktt::ArgumentId>{atomInfoPrecompId, atomInfoXId, atomInfoYId, atomInfoZ2Id, atomInfoWId,
        numberOfAtomsId, gridSpacingId, zIndexId, energyGridId});
    tuner.SetArguments(referenceDefinition, std::vector<ktt::ArgumentId>{atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId});

    // Multiply work-group size in dimensions x and y by two parameters that follow (effectively setting work-group size to parameters' values)
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{4, 8, 16, 32});
    tuner.SetThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.SetThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);

    // Add additional tuning parameters
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1, 2});

    // Using vectorized SoA only makes sense when vectors are longer than 1
    auto vectorizedSoA = [](const std::vector<uint64_t>& vector) {return vector.at(0) > 1 || vector.at(1) != 2;};
    tuner.AddConstraint(kernel, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);
    // Ensure sufficient parallelism
    auto par = [](const std::vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
    tuner.AddConstraint(kernel, std::vector<std::string>{"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.SetThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
        ktt::ModifierAction::Divide);

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    tuner.SetReferenceKernel(energyGridId, referenceKernel, ktt::KernelConfiguration());

    // Set tuning manipulator, which implements custom method for launching the kernel
    tuner.SetLauncher(kernel, [definition, gridSize, atoms, gridSpacing, atomInfoPrecompId, atomInfoZ2Id, &atomInfoZ, atomInfoZ2,
        atomInfoPrecomp, zIndexId](ktt::ComputeInterface& interface) mutable
    {
        // Get kernel data
        ktt::DimensionVector globalSize = interface.GetCurrentGlobalSize(definition);
        const ktt::DimensionVector& localSize = interface.GetCurrentLocalSize(definition);
        const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();

        globalSize.SetSizeZ(1);

        // Iterate over slices
        for (int i = 0; i < gridSize; ++i)
        {
            // Perform precomputation for 2D kernel
            const float z = gridSpacing * static_cast<float>(i);

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "USE_SOA") == 0)
            {
                for (int j = 0; j < atoms; ++j)
                {
                    atomInfoPrecomp[j * 4 + 2] = (z - atomInfoZ[j]) * (z - atomInfoZ[j]);
                }

                const auto transferId = interface.UpdateBuffer(atomInfoPrecompId, interface.GetDefaultQueue(), atomInfoPrecomp.data(),
                    atomInfoPrecomp.size() * sizeof(float));
                interface.WaitForTransferAction(transferId);
            }
            else
            {
                for (int j = 0; j < atoms; j++)
                {
                    atomInfoZ2[j] = (z - atomInfoZ[j]) * (z - atomInfoZ[j]);
                }

                const auto transferId = interface.UpdateBuffer(atomInfoZ2Id, interface.GetDefaultQueue(), atomInfoZ2.data(),
                    atomInfoZ2.size() * sizeof(float));
                interface.WaitForTransferAction(transferId);
            }

            interface.UpdateScalarArgument(zIndexId, &i);
            interface.RunKernel(definition, globalSize, localSize);
        }
    });
    
    const std::vector<ktt::KernelResult> results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "CoulombSum3dIterativeOutput", ktt::OutputFormat::JSON);

    return 0;
}
