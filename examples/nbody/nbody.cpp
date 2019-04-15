#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#define USE_CUDA 0
#define USE_PROFILING 0

#if USE_CUDA == 0
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/nbody/nbody_kernel1.cl"
        #define KTT_REFERENCE_KERNEL_FILE "../examples/nbody/nbody_reference_kernel.cl"
    #else
        #define KTT_KERNEL_FILE "../../examples/nbody/nbody_kernel1.cl"
        #define KTT_REFERENCE_KERNEL_FILE "../../examples/nbody/nbody_reference_kernel.cl"
    #endif
#else
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/nbody/nbody_kernel1.cu"
        #define KTT_REFERENCE_KERNEL_FILE "../examples/nbody/nbody_reference_kernel.cu"
    #else
        #define KTT_KERNEL_FILE "../../examples/nbody/nbody_kernel1.cu"
        #define KTT_REFERENCE_KERNEL_FILE "../../examples/nbody/nbody_reference_kernel.cu"
    #endif
#endif

int main(int argc, char** argv)
{
    // Initialize platform and device index
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
    const int numberOfBodies = 32 * 1024;
    // Total NDRange size matches number of grid points
    const ktt::DimensionVector ndRangeDimensions(numberOfBodies);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(64);

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
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 20.0f);

    for (int i = 0; i < numberOfBodies; i++)
    {
        oldPosX.at(i) = distribution(engine);
        oldPosY.at(i) = distribution(engine);
        oldPosZ.at(i) = distribution(engine);
        bodyMass.at(i) = distribution(engine);

        oldVelX.at(i) = distribution(engine);
        oldVelY.at(i) = distribution(engine);
        oldVelZ.at(i) = distribution(engine);

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
#if USE_CUDA == 0
    ktt::Tuner tuner(platformIndex, deviceIndex);
    tuner.setCompilerOptions("-cl-fast-relaxed-math");
#else
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.setCompilerOptions("-use_fast_math");
  #if USE_PROFILING == 1
    printf("Executing with profiling switched ON.\n");
    tuner.setKernelProfiling(true);
  #endif
#endif
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);
    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "nbody_kernel", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "nbody_kernel", ndRangeDimensions, referenceWorkGroupDimensions);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {64, 128, 256, 512});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", {1, 2, 4, 8});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR", ktt::ModifierAction::Divide);
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", {0, 1, 2, 4, 8, 16, 32});
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", {0, 1, 2, 4, 8, 16, 32});
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0, 1});
#else
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0});
#endif
    tuner.addParameter(kernelId, "USE_SOA", {0, 1});
    tuner.addParameter(kernelId, "LOCAL_MEM", {0, 1});
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4, 8, 16});
#else
    tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4});
#endif

    // Add all arguments utilized by kernels
    ktt::ArgumentId oldBodyInfoId = tuner.addArgumentVector(oldBodyInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldPosXId = tuner.addArgumentVector(oldPosX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldPosYId = tuner.addArgumentVector(oldPosY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldPosZId = tuner.addArgumentVector(oldPosZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId massId = tuner.addArgumentVector(bodyMass, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId newBodyInfoId = tuner.addArgumentVector(newBodyInfo, ktt::ArgumentAccessType::WriteOnly);

    ktt::ArgumentId oldVelId = tuner.addArgumentVector(oldBodyVel, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldVelXId = tuner.addArgumentVector(oldVelX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldVelYId = tuner.addArgumentVector(oldVelY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId oldVelZId = tuner.addArgumentVector(oldVelZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId newBodyVelId = tuner.addArgumentVector(newBodyVel, ktt::ArgumentAccessType::WriteOnly);

    ktt::ArgumentId deltaTimeId = tuner.addArgumentScalar(timeDelta);
    ktt::ArgumentId dampingId = tuner.addArgumentScalar(damping);
    ktt::ArgumentId softeningSqrId = tuner.addArgumentScalar(softeningSqr);
    ktt::ArgumentId numberOfBodiesId = tuner.addArgumentScalar(numberOfBodies);

    // Add conditions
    auto lteq = [](const std::vector<size_t>& vector) {return vector.at(0) <= vector.at(1);};
    tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, lteq);
    auto lteq256 = [](const std::vector<size_t>& vector) {return vector.at(0) * vector.at(1) <= 256;};
    tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, lteq256);
    auto vectorizedSoA = [](const std::vector<size_t>& vector) {return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1);};
    tuner.addConstraint(kernelId, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{deltaTimeId,
        oldBodyInfoId, oldPosXId, oldPosYId, oldPosZId, massId, newBodyInfoId, // position
        oldVelId, oldVelXId, oldVelYId, oldVelZId, newBodyVelId, // velocity
        dampingId, softeningSqrId, numberOfBodiesId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{deltaTimeId, oldBodyInfoId, newBodyInfoId, oldVelId, newBodyVelId,
        dampingId, softeningSqrId});

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{newBodyVelId,
        newBodyInfoId});
  
    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "nbody_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
