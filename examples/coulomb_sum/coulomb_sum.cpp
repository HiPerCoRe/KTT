#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    auto kernelFile = std::string("../examples/coulomb_sum/coulomb_sum_kernel.cl");
    auto referenceKernelFile = std::string("../examples/coulomb_sum/coulomb_sum_reference_kernel.cl");

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string{ argv[1] });
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string{ argv[2] });
            if (argc >= 4)
            {
                kernelFile = std::string{ argv[3] };
                if (argc >= 5)
                {
                    referenceKernelFile = std::string{ argv[4] };
                }
            }
        }
    }

    // Declare kernel parameters
    const ktt::DimensionVector ndRangeDimensions(512, 512, 1);
    const ktt::DimensionVector workGroupDimensions(1, 1, 1);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16, 1);
    // Total NDRange size matches number of grid points
    const size_t numberOfGridPoints = std::get<0>(ndRangeDimensions) * std::get<1>(ndRangeDimensions);
    // Used for generating random test data
    const float upperBoundary = 20.0f; 
    // If higher than 4k, computations with constant memory enabled will be invalid on many devices due to constant memory capacity limit
    const int numberOfAtoms = 4096;

    // Declare data variables
    float gridSpacing;
    std::vector<float> atomInfo(4 * numberOfAtoms);
    std::vector<float> atomInfoX(numberOfAtoms);
    std::vector<float> atomInfoY(numberOfAtoms);
    std::vector<float> atomInfoZ(numberOfAtoms);
    std::vector<float> atomInfoW(numberOfAtoms);
    std::vector<float> energyGrid(numberOfGridPoints, 0.0f);

    // Initialize data
    srand(static_cast<unsigned int>(time(0)));
    gridSpacing = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < numberOfAtoms; i++)
    {
        atomInfoX.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        atomInfoY.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        atomInfoZ.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        atomInfoW.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);

        atomInfo.at((4 * i)) = atomInfoX.at(i);
        atomInfo.at((4 * i) + 1) = atomInfoY.at(i);
        atomInfo.at((4 * i) + 2) = atomInfoZ.at(i);
        atomInfo.at((4 * i) + 3) = atomInfoW.at(i);
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("directCoulombSum"), ndRangeDimensions, workGroupDimensions);
    size_t referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, std::string("directCoulombSumReference"), ndRangeDimensions,
        referenceWorkGroupDimensions);

    // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers
    tuner.addParameter(kernelId, std::string("INNER_UNROLL_FACTOR"), std::vector<size_t>{ 1, 2, 4, 8 });
    tuner.addParameter(kernelId, std::string("USE_CONSTANT_MEMORY"), std::vector<size_t>{ 0, 1 });
    tuner.addParameter(kernelId, std::string("VECTOR_TYPE"), std::vector<size_t>{ 1, 2, 4, 8 });
    tuner.addParameter(kernelId, std::string("USE_SOA"), std::vector<size_t>{ 0, 1, 2 });

    // Using vectorized SoA only makes sense when vectors are longer than 1
    auto vectorizedSoA = [](std::vector<size_t> vector) { return vector.at(0) > 1 || vector.at(1) != 2; }; 
    tuner.addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{ "VECTOR_TYPE", "USE_SOA" });

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    tuner.addParameter(kernelId, std::string("OUTER_UNROLL_FACTOR"), std::vector<size_t>{ 1, 2, 4, 8 }, ktt::ThreadModifierType::Global,
        ktt::ThreadModifierAction::Divide, ktt::Dimension::X);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, std::string("WORK_GROUP_SIZE_X"), std::vector<size_t>{ 4, 8, 16, 32 }, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);
    tuner.addParameter(kernelId, std::string("WORK_GROUP_SIZE_Y"), std::vector<size_t>{ 1, 2, 4, 8, 16, 32 }, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::Y);

    // Add all arguments utilized by kernels
    size_t atomInfoId = tuner.addArgument(atomInfo, ktt::ArgumentMemoryType::ReadOnly);
    size_t atomInfoXId = tuner.addArgument(atomInfoX, ktt::ArgumentMemoryType::ReadOnly);
    size_t atomInfoYId = tuner.addArgument(atomInfoY, ktt::ArgumentMemoryType::ReadOnly);
    size_t atomInfoZId = tuner.addArgument(atomInfoZ, ktt::ArgumentMemoryType::ReadOnly);
    size_t atomInfoWId = tuner.addArgument(atomInfoW, ktt::ArgumentMemoryType::ReadOnly);
    size_t numberOfAtomsId = tuner.addArgument(numberOfAtoms);
    size_t gridSpacingId = tuner.addArgument(gridSpacing);
    size_t energyGridId = tuner.addArgument(energyGrid, ktt::ArgumentMemoryType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId,
        std::vector<size_t>{ atomInfoId, atomInfoXId, atomInfoYId, atomInfoZId, atomInfoWId, numberOfAtomsId, gridSpacingId, energyGridId });
    tuner.setKernelArguments(referenceKernelId, std::vector<size_t>{ atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId });

    // Set search method to random search, only 10% of all configurations will be explored.
    tuner.setSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double>{ 0.1 });

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterValue>{}, std::vector<size_t>{ energyGridId });

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);

    return 0;
}
