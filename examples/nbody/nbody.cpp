#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>

#include "../../include/ktt.h"

int main(int argc, char** argv)
{
    // Initialize platform and device index
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    auto kernelFile = std::string("../examples/nbody/nbody_kernel1.cl");
    auto referenceKernelFile = std::string("../examples/nbody/nbody_reference_kernel.cl");

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

    // Used for generating random test data
    const float upperBoundary = 20.0f;
    const int numberOfBodies = 8192;
	
	 // Declare kernel parameters
    // Total NDRange size matches number of grid points
    ktt::DimensionVector ndRangeDimensions(numberOfBodies, 1, 1);
    ktt::DimensionVector workGroupDimensions(1, 1, 1);
    ktt::DimensionVector referenceWorkGroupDimensions(1, 1, 1);

	 // Declare data variables
	float timeDelta = 0.001f;
	float damping = 0.5f;
	float softeningSqr = 0.1f * 0.1f;
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
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < numberOfBodies; i++)
    {
        oldPosX.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        oldPosY.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        oldPosZ.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyMass.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
		
		oldVelX.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        oldVelY.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        oldVelZ.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);

        oldBodyInfo.at((4 * i)) = oldPosX.at(i);
        oldBodyInfo.at((4 * i) + 1) = oldPosY.at(i);
        oldBodyInfo.at((4 * i) + 2) = oldPosZ.at(i);
        oldBodyInfo.at((4 * i) + 3) = bodyMass.at(i);
		
		oldBodyVel.at((4 * i)) = oldVelX.at(i);
        oldBodyVel.at((4 * i) + 1) = oldVelY.at(i);
        oldBodyVel.at((4 * i) + 2) = oldVelZ.at(i);
        oldBodyVel.at((4 * i) + 3) = 0.f;
    }
	
	// Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

	// Add two kernels to tuner, one of the kernels acts as reference kernel
    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("nbody_kernel"), ndRangeDimensions, workGroupDimensions);
    size_t referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, std::string("nbody_kernel"), ndRangeDimensions,
        referenceWorkGroupDimensions);

	 // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, std::string("WORK_GROUP_SIZE_X"), std::vector<size_t>{ 32, 64, 128, 256, 512}, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);
	tuner.addParameter(kernelId, std::string("OUTER_UNROLL_FACTOR"), std::vector<size_t>{ 1, 2, 4, 8, 16, 32 }, ktt::ThreadModifierType::Global,
        ktt::ThreadModifierAction::Divide, ktt::Dimension::X);
	tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", { 1, 2, 4, 8, 16, 32, 64, 128, 256 });
	tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", { 1, 2, 4, 8, 16, 32 });
	tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", { 0, 1 });
	tuner.addParameter(kernelId, "USE_SOA", { 0, 1 });
	tuner.addParameter(kernelId, "LOCAL_MEM", { 0, 1 });
	tuner.addParameter(kernelId, std::string("VECTOR_TYPE"), std::vector<size_t>{ 1, 2, 4, 8, 16 });
		
		
	 // Add all arguments utilized by kernels
    size_t oldBodyInfoId = tuner.addArgument(oldBodyInfo, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldPosXId = tuner.addArgument(oldPosX, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldPosYId = tuner.addArgument(oldPosY, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldPosZId = tuner.addArgument(oldPosZ, ktt::ArgumentMemoryType::ReadOnly);
	size_t massId = tuner.addArgument(bodyMass, ktt::ArgumentMemoryType::ReadOnly);
    size_t newBodyInfoId = tuner.addArgument(newBodyInfo, ktt::ArgumentMemoryType::WriteOnly);
	
    size_t oldVelId = tuner.addArgument(oldBodyVel, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldVelXId = tuner.addArgument(oldVelX, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldVelYId = tuner.addArgument(oldVelY, ktt::ArgumentMemoryType::ReadOnly);
	size_t oldVelZId = tuner.addArgument(oldVelZ, ktt::ArgumentMemoryType::ReadOnly);
	size_t newBodyVelId = tuner.addArgument(newBodyVel, ktt::ArgumentMemoryType::WriteOnly);
	
    size_t deltaTimeId = tuner.addArgument(timeDelta);
    size_t dampingId = tuner.addArgument(damping);
	size_t softeningSqrId = tuner.addArgument(softeningSqr);
	size_t numberOfBodiesId = tuner.addArgument(numberOfBodies);
	
	// Add conditions
	auto lteq = [](std::vector<size_t> vector) { return vector.at(0) <= vector.at(1); };
	tuner.addConstraint(kernelId, lteq, { "INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR" } );
	auto lteq256 = [](std::vector<size_t> vector) { return vector.at(0) * vector.at(1) <= 256; };
	tuner.addConstraint(kernelId, lteq, { "INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2" } );
	// Using vectorized SoA only makes sense when vectors are longer than 1
    auto vectorizedSoA = [](std::vector<size_t> vector) { return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1); };
    tuner.addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{ "VECTOR_TYPE", "USE_SOA" });


	// Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId,
        std::vector<size_t>{ deltaTimeId, 
		oldBodyInfoId, oldPosXId, oldPosYId, oldPosZId, massId, newBodyInfoId, // position
		oldVelId, oldVelXId, oldVelYId, oldVelZId, newBodyVelId, // velocity
		dampingId, softeningSqrId, numberOfBodiesId });
    tuner.setKernelArguments(referenceKernelId, 
		std::vector<size_t>{ deltaTimeId, oldBodyInfoId, newBodyInfoId, oldVelId, newBodyVelId, dampingId, softeningSqrId });

	// Set search method to random search, only 1% of all configurations will be explored.
    //tuner.setSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double>{ 0.01 });

	  // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001f);

	 // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterValue>{}, std::vector<size_t>{ newBodyVelId, newBodyInfoId });
  
    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);

    return 0;

}
