#pragma once

#include "tuner_api.h"

class tunableCoulomb : public ktt::TuningManipulator {
    ktt::Tuner *tuner;
    int atoms;
    int gridSize;
    float gridSpacing;
    std::vector<float> atomInfo;
    std::vector<float> atomInfoPrecomp;
    std::vector<float> atomInfoX;
    std::vector<float> atomInfoY;
    std::vector<float> atomInfoZ;
    std::vector<float> atomInfoZ2;
    std::vector<float> atomInfoW;
    std::vector<float> energyGrid;

    size_t kernelId;
    size_t referenceKernelId;
    size_t atomInfoId;
    size_t atomInfoPrecompId;
    size_t atomInfoXId;
    size_t atomInfoYId;
    size_t atomInfoZId;
    size_t atomInfoZ2Id;
    size_t atomInfoWId;
    size_t numberOfAtomsId;
    size_t gridSpacingId;
    size_t zIndexId;
    size_t energyGridId;

    void initWithRandomData() {
        energyGrid.assign(gridSize*gridSize*gridSize, 0.0f);
        atomInfoX.resize(atoms);
        atomInfoY.resize(atoms);
        atomInfoZ.resize(atoms);
        atomInfoZ2.resize(atoms);
        atomInfoW.resize(atoms);
        atomInfo.resize(atoms*4);
        atomInfoPrecomp.resize(atoms*4);
        gridSpacing = 0.5f;
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, 40.0f);
        for (int i = 0; i < atoms; i++) {
            atomInfoX.at(i) = distribution(engine);
            atomInfoY.at(i) = distribution(engine);
            atomInfoZ.at(i) = distribution(engine);
            atomInfoW.at(i) = distribution(engine)/40.0f;

            atomInfo.at((4 * i)) = atomInfoX.at(i);
            atomInfo.at((4 * i) + 1) = atomInfoY.at(i);
            atomInfo.at((4 * i) + 2) = atomInfoZ.at(i);
            atomInfo.at((4 * i) + 3) = atomInfoW.at(i);

            // do not store z, it will be rewritten anyway
            atomInfoPrecomp.at((4 * i)) = atomInfoX.at(i);
            atomInfoPrecomp.at((4 * i) + 1) = atomInfoY.at(i);
            atomInfoPrecomp.at((4 * i) + 3) = atomInfoW.at(i);
        }
    }
public:
/* 
    Constructor creates internal structures and setups tuning environment */
    tunableCoulomb(ktt::Tuner *tuner, int gridSize, int atoms) {
        this->tuner = tuner;

        // configure input size and create random input
        this->atoms = atoms;
        this->gridSize = gridSize;
        initWithRandomData();

        // create kernel
        const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
        const ktt::DimensionVector workGroupDimensions(1, 1, 1);
        const ktt::DimensionVector referenceWorkGroupDimensions(16, 16, 1);
        kernelId = tuner->addKernelFromFile("../examples/coulomb_sum_3d_iterative/coulomb_sum_3d_iterative_kernel.cl", "directCoulombSum", ndRangeDimensions, workGroupDimensions);
        referenceKernelId = tuner->addKernelFromFile("../examples/coulomb_sum_3d_iterative/coulomb_sum_3d_iterative_reference_kernel.cl", "directCoulombSumReference", ndRangeDimensions, referenceWorkGroupDimensions);

        // create input/output in tuner
        atomInfoId = tuner->addArgument(atomInfo, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoPrecompId = tuner->addArgument(atomInfoPrecomp, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoXId = tuner->addArgument(atomInfoX, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoYId = tuner->addArgument(atomInfoY, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoZId = tuner->addArgument(atomInfoZ, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoZ2Id = tuner->addArgument(atomInfoZ2, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        atomInfoWId = tuner->addArgument(atomInfoW, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly);
        numberOfAtomsId = tuner->addArgument(atoms);
        gridSpacingId = tuner->addArgument(gridSpacing);
        int zIndex = 0;
        zIndexId = tuner->addArgument(zIndex);
        energyGridId = tuner->addArgument(energyGrid, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadWrite);
        tuner->setKernelArguments(kernelId, std::vector<size_t>{ atomInfoPrecompId, atomInfoXId, atomInfoYId, atomInfoZ2Id, atomInfoWId, numberOfAtomsId, gridSpacingId, zIndexId, energyGridId });
        tuner->setKernelArguments(referenceKernelId, std::vector<size_t>{ atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId });

        // create parameter space

        // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
        tuner->addParameter(kernelId, std::string("WORK_GROUP_SIZE_X"), std::vector<size_t>{ /*4, 8, 16,*/ 32 }, ktt::ThreadModifierType::Local, ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);
        tuner->addParameter(kernelId, std::string("WORK_GROUP_SIZE_Y"), std::vector<size_t>{ 1, 2, 4, 8, 16, 32 }, ktt::ThreadModifierType::Local, ktt::ThreadModifierAction::Multiply, ktt::Dimension::Y);
        tuner->addParameter(kernelId, std::string("INNER_UNROLL_FACTOR"), std::vector<size_t>{ 0, 1, 2, 4, 8, 16, 32 });
        tuner->addParameter(kernelId, std::string("USE_CONSTANT_MEMORY"), std::vector<size_t>{ 0, 1 });
        tuner->addParameter(kernelId, std::string("VECTOR_TYPE"), std::vector<size_t>{ 1, 2, 4, 8 });
        tuner->addParameter(kernelId, std::string("USE_SOA"), std::vector<size_t>{ 0, 1, 2 });
        // Using vectorized SoA only makes sense when vectors are longer than 1
        auto vectorizedSoA = [](std::vector<size_t> vector) { return vector.at(0) > 1 || vector.at(1) != 2; };
        tuner->addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{ "VECTOR_TYPE", "USE_SOA" });
        // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
        tuner->addParameter(kernelId, std::string("OUTER_UNROLL_FACTOR"), std::vector<size_t>{ 1, 2, 4, 8 }, ktt::ThreadModifierType::Global, ktt::ThreadModifierAction::Divide, ktt::Dimension::X);
        
         // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
        tuner->setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);
   
        //tuner->setSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double> { 0.01 });
 
        // set reference kernel
        tuner->setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterValue>{}, std::vector<size_t>{ energyGridId });
    }

/*
    launchComputation is responsible for actual execution of tuned kernel */
    virtual void launchComputation(const size_t kernelId) override {
        // get kernel data
        ktt::DimensionVector globalSize = getCurrentGlobalSize(kernelId);
        ktt::DimensionVector localSize = getCurrentLocalSize(kernelId);
        std::vector<ktt::ParameterValue> parameterValues = getCurrentConfiguration();

        std::get<2>(globalSize) = 1;

        // iterate over slices
        for (int i = 0; i < gridSize; i++) {
            // perform precomputation for 2D kernel
            float z = gridSpacing * float(i);
            if (getParameterValue("USE_SOA", parameterValues) == 0) {
                for (int j = 0; j < atoms; j++)
                    atomInfoPrecomp[j*4+2] = (z-atomInfoZ[j])*(z-atomInfoZ[j]);
                updateArgumentVector(atomInfoPrecompId, atomInfoPrecomp.data());
            }
            else {
                for (int j = 0; j < atoms; j++)
                    atomInfoZ2[j] = (z-atomInfoZ[j])*(z-atomInfoZ[j]);
                updateArgumentVector(atomInfoZ2Id, atomInfoZ2.data());
            }
            updateArgumentScalar(zIndexId, &i);
        
            runKernel(kernelId, globalSize, localSize);
        }
    }

/*
    perform tuning and store results */
    void tune() {
        tuner->tuneKernel(kernelId);
        tuner->printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
        tuner->printResult(kernelId, std::string("coulomb_sum_3d_iterative_output.csv"), ktt::PrintFormat::CSV);
    }

/*
    simple utility methods */
    size_t getKernelId() const {
        return kernelId;
    }
};

