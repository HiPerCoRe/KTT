#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../include/ktt.h"

int main(int argc, char** argv)
{
    // Initialize platform and device index
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    auto kernelFile = std::string("../examples/nbody/nbody_kernel.cl");
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
	
	 // Declare kernel parameters
    // Total NDRange size matches number of grid points
    ktt::DimensionVector ndRangeDimensions(512, 512, 1);
    ktt::DimensionVector workGroupDimensions(1, 1, 1);
    ktt::DimensionVector referenceWorkGroupDimensions(1, 1, 1);
	// Used for generating random test data
    const float upperBoundary = 20.0f; 
    const int numberOfBodies = 4096;

	 // Declare data variables
	float timeDelta = 0.001f;
	float damping = 0.5f;
    std::vector<float> oldBodyInfo(4 * numberOfBodies);
	std::vector<float> bodyPosX(numberOfBodies);
    std::vector<float> bodyPosY(numberOfBodies);
    std::vector<float> bodyPosZ(numberOfBodies);
    std::vector<float> bodyMass(numberOfBodies);
	
	std::vector<float> newBodyInfo(4 * numberOfBodies, 0.f);
	
	std::vector<float> bodyVel(4 * numberOfBodies);
	std::vector<float> bodyVelX(numberOfBodies);
    std::vector<float> bodyVelY(numberOfBodies);
    std::vector<float> bodyVelZ(numberOfBodies);
	
	// Initialize data
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < numberOfBodies; i++)
    {
        bodyPosX.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyPosY.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyPosZ.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyMass.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
		
		bodyVelX.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyVelY.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);
        bodyVelZ.at(i) = static_cast<float>(rand()) / (RAND_MAX / upperBoundary);

        oldBodyInfo.at((4 * i)) = bodyPosX.at(i);
        oldBodyInfo.at((4 * i) + 1) = bodyPosY.at(i);
        oldBodyInfo.at((4 * i) + 2) = bodyPosZ.at(i);
        oldBodyInfo.at((4 * i) + 3) = bodyMass.at(i);
		
		bodyVel.at((4 * i)) = bodyVelX.at(i);
        bodyVel.at((4 * i) + 1) = bodyVelY.at(i);
        bodyVel.at((4 * i) + 2) = bodyVelZ.at(i);
        bodyVel.at((4 * i) + 3) = 0.f;
    }
	
	// Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

	// Add two kernels to tuner, one of the kernels acts as reference kernel
    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("integrateBodies"), ndRangeDimensions, workGroupDimensions);
    size_t referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, std::string("nbody_kern"), ndRangeDimensions,
        referenceWorkGroupDimensions);

	 // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, std::string("WORK_GROUP_SIZE_X"), std::vector<size_t>{ 4, 8, 16, 32 }, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);
    // tuner.addParameter(kernelId, std::string("WORK_GROUP_SIZE_Y"), std::vector<size_t>{ 1, 2, 4, 8, 16, 32 }, ktt::ThreadModifierType::Local,
        // ktt::ThreadModifierAction::Multiply, ktt::Dimension::Y);

	 // Add all arguments utilized by kernels
    size_t deltaTimeId = tuner.addArgument(timeDelta);
    size_t oldBodyInfoId = tuner.addArgument(oldBodyInfo, ktt::ArgumentMemoryType::ReadOnly);
    size_t newBodyInfoId = tuner.addArgument(newBodyInfo, ktt::ArgumentMemoryType::WriteOnly);
    size_t bodyVelId = tuner.addArgument(bodyVel, ktt::ArgumentMemoryType::ReadWrite);
    size_t dampingId = tuner.addArgument(damping);

	// Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId,
        std::vector<size_t>{ deltaTimeId, oldBodyInfoId, newBodyInfoId, bodyVelId, dampingId });
    tuner.setKernelArguments(referenceKernelId, 
		std::vector<size_t>{ deltaTimeId, oldBodyInfoId, newBodyInfoId, bodyVelId, dampingId });

	  // Set search method to random search, only 10% of all configurations will be explored.
    // tuner.setSearchMethod(kernelId, ktt::SearchMethod::RandomSearch, std::vector<double>{ 0.2 });

	  // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

	 // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterValue>{}, std::vector<size_t>{ newBodyInfoId });
  
    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);

    return 0;

}

/*
// Implementation of reference class interface for result validation
class SimpleReferenceClass : public ktt::ReferenceClass
{
public:
    SimpleReferenceClass(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& result, const size_t resultArgumentId) :
        a(a),
        b(b),
        result(result),
        resultArgumentId(resultArgumentId)
    {}

    virtual void computeResult() override
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            result.at(i) = a.at(i) + b.at(i);
        }
    }

    virtual void* getData(const size_t argumentId) const override
    {
        if (argumentId == resultArgumentId)
        {
            return (void*)result.data();
        }
        throw std::runtime_error("No result available for specified argument id");
    }

    virtual ktt::ArgumentDataType getDataType(const size_t argumentId) const override
    {
        return ktt::ArgumentDataType::Float;
    }

    virtual size_t getDataSizeInBytes(const size_t argumentId) const override
    {
        return result.size() * sizeof(float);
    }

private:
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> result;
    size_t resultArgumentId;
};

int main(int argc, char** argv)
{
    // Initialize platform and device index
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    auto kernelFile = std::string("../examples/nbody/nbody_kernel.cl");

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string{ argv[1] });
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string{ argv[2] });
            if (argc >= 4)
            {
                kernelFile = std::string{ argv[3] };
            }
        }
    }

    // Declare kernel parameters
    const size_t numberOfElements = 1024 * 1024;
    ktt::DimensionVector ndRangeDimensions(numberOfElements, 1, 1);
    ktt::DimensionVector workGroupDimensions(256, 1, 1);

    // Declare data variables
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; i++)
    {
        a.at(i) = static_cast<float>(i);
        b.at(i) = static_cast<float>(i + 1);
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add new kernel to tuner, specify kernel name, NDRange dimensions and work-group dimensions
    size_t kernelId = tuner.addKernelFromFile(kernelFile, std::string("nBodyKernel"), ndRangeDimensions, workGroupDimensions);

    // Add new arguments to tuner, argument data is copied from std::vector containers
    size_t aId = tuner.addArgument(a, ktt::ArgumentMemoryType::ReadOnly);
    size_t bId = tuner.addArgument(b, ktt::ArgumentMemoryType::ReadOnly);
    size_t resultId = tuner.addArgument(result, ktt::ArgumentMemoryType::WriteOnly);

    // Set kernel arguments by providing corresponding argument ids returned by addArgument() method, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<size_t>{ aId, bId, resultId });

    // Set reference class, which implements C++ version of kernel computation in order to validate results provided by kernel,
    // provide list of arguments which will be validated
    tuner.setReferenceClass(kernelId, std::make_unique<SimpleReferenceClass>(a, b, result, resultId), std::vector<size_t>{ resultId });

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);
    return 0;
}
*/