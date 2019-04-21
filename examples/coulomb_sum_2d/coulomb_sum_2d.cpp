#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/coulomb_sum_2d/coulomb_sum_2d_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/coulomb_sum_2d/coulomb_sum_2d_reference_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/coulomb_sum_2d/coulomb_sum_2d_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/coulomb_sum_2d/coulomb_sum_2d_reference_kernel.cl"
#endif

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;
    std::string referenceKernelFile = KTT_REFERENCE_KERNEL_FILE;

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
    const ktt::DimensionVector ndRangeDimensions(512, 512);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);
    // Total NDRange size matches number of grid points
    const size_t numberOfGridPoints = ndRangeDimensions.getSizeX() * ndRangeDimensions.getSizeY();
    // If higher than 4k, computations with constant memory enabled will be invalid on many devices due to constant memory capacity limit
    const int numberOfAtoms = 4000;

    // Declare data variables
    float gridSpacing = 0.5f;
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

    for (int i = 0; i < numberOfAtoms; i++)
    {
        atomInfoX.at(i) = distribution(engine);
        atomInfoY.at(i) = distribution(engine);
        atomInfoZ.at(i) = distribution(engine);
        atomInfoW.at(i) = distribution(engine) / 40.0f;

        atomInfo.at((4 * i)) = atomInfoX.at(i);
        atomInfo.at((4 * i) + 1) = atomInfoY.at(i);
        atomInfo.at((4 * i) + 2) = atomInfoZ.at(i);
        atomInfo.at((4 * i) + 3) = atomInfoW.at(i);
    }

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);
    tuner.setCompilerOptions("-cl-fast-relaxed-math");
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "directCoulombSum", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "directCoulombSumReference", ndRangeDimensions,
        referenceWorkGroupDimensions);

    // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", std::vector<size_t>{0, 1, 2, 4, 8, 16, 32});
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", std::vector<size_t>{0, 1});
    tuner.addParameter(kernelId, "VECTOR_TYPE", std::vector<size_t>{1, 2, 4, 8});
    tuner.addParameter(kernelId, "USE_SOA", std::vector<size_t>{0, 1, 2});

    // Using vectorized SoA only makes sense when vectors are longer than 1
    auto vectorizedSoA = [](const std::vector<size_t>& vector) {return vector.at(0) > 1 || vector.at(1) != 2;}; 
    tuner.addConstraint(kernelId, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", std::vector<size_t>{1, 2, 4, 8});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR", ktt::ModifierAction::Divide);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", std::vector<size_t>{4, 8, 16, 32});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", std::vector<size_t>{1, 2, 4, 8, 16, 32});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);

    // Add all arguments utilized by kernels
    ktt::ArgumentId atomInfoId = tuner.addArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoXId = tuner.addArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoYId = tuner.addArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoZId = tuner.addArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoWId = tuner.addArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId numberOfAtomsId = tuner.addArgumentScalar(numberOfAtoms);
    ktt::ArgumentId gridSpacingId = tuner.addArgumentScalar(gridSpacing);
    ktt::ArgumentId energyGridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{atomInfoId, atomInfoXId, atomInfoYId, atomInfoZId, atomInfoWId, numberOfAtomsId,
        gridSpacingId, energyGridId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId});

    // Set search method to random search
    tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, std::vector<double>{});

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{energyGridId});

    // Launch kernel tuning, end after exploring 10% of configurations
    tuner.tuneKernel(kernelId, std::make_unique<ktt::ConfigurationFraction>(0.1));

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "coulomb_sum_2d_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
