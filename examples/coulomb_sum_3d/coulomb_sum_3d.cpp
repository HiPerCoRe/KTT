#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tuner_api.h"

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    auto kernelFile = std::string("../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cl");
    auto referenceKernelFile = std::string("../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cl");

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
    const int gridSize = 256;
    const int atoms = 4000;
    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions(1, 1, 1);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16, 1);

    // Declare data variables
    float gridSpacing;
    std::vector<float> atomInfoX(atoms);
    std::vector<float> atomInfoY(atoms);
    std::vector<float> atomInfoZ(atoms);
    std::vector<float> atomInfoW(atoms);
    std::vector<float> atomInfo(atoms*4);
    std::vector<float> energyGrid(gridSize*gridSize*gridSize, 0.0f);

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
        atomInfoW.at(i) = distribution(engine)/40.0f;
        atomInfo.at(i*4) = atomInfoX.at(i);
        atomInfo.at(i*4+1) = atomInfoY.at(i);
        atomInfo.at(i*4+2) = atomInfoZ.at(i);
        atomInfo.at(i*4+3) = atomInfoW.at(i);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex);

    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("directCoulombSum"), ndRangeDimensions, workGroupDimensions);
    size_t referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, std::string("directCoulombSumReference"), ndRangeDimensions,
        referenceWorkGroupDimensions);

    size_t aiId = tuner.addArgument(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    size_t aixId = tuner.addArgument(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    size_t aiyId = tuner.addArgument(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    size_t aizId = tuner.addArgument(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    size_t aiwId = tuner.addArgument(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    size_t aId = tuner.addArgument(atoms);
    size_t gsId = tuner.addArgument(gridSpacing);
    size_t gridId = tuner.addArgument(energyGrid, ktt::ArgumentAccessType::WriteOnly);

    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 16, 32 }, 
        ktt::ThreadModifierType::Local, 
        ktt::ThreadModifierAction::Multiply, 
        ktt::Dimension::X);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 1, 2, 4, 8 }, 
        ktt::ThreadModifierType::Local, 
        ktt::ThreadModifierAction::Multiply, 
        ktt::Dimension::Y);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Z", { 1 });
    tuner.addParameter(kernelId, "Z_ITERATIONS", { 1, 2, 4, 8, 16, 32 },
        ktt::ThreadModifierType::Global,
        ktt::ThreadModifierAction::Divide,
        ktt::Dimension::Z);
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", { 0, 1, 2, 4, 8, 16, 32 });
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", { 0, 1 });
    tuner.addParameter(kernelId, "USE_SOA", { 0, 1 });
    tuner.addParameter(kernelId, "VECTOR_SIZE", { 1, 2 , 4, 8, 16 });

    auto lt = [](std::vector<size_t> vector) { return vector.at(0) < vector.at(1); };
    tuner.addConstraint(kernelId, lt, { "INNER_UNROLL_FACTOR", "Z_ITERATIONS" } );
    auto vec = [](std::vector<size_t> vector) { return vector.at(0) || vector.at(1) == 1; };
    tuner.addConstraint(kernelId, vec, { "USE_SOA", "VECTOR_SIZE" } );

    tuner.setKernelArguments(kernelId, std::vector<size_t>{ aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridId });
    tuner.setKernelArguments(referenceKernelId, std::vector<size_t>{ aiId, aId, gsId, gridId });

    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterValue>{}, std::vector<size_t>{ gridId });
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    //tuner.setSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double> { 0.1 });

    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("coulomb_sum_3d_output.csv"), ktt::PrintFormat::CSV);

    return 0;
}
