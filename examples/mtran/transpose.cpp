#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#define USE_CUDA 0
#define USE_PROFILING 0

#if USE_CUDA == 0
  #if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/mtran/mtran_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/mtran/mtran_reference_kernel.cl"
  #else
    #define KTT_KERNEL_FILE "../../examples/mtran/mtran_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/mtran/mtran_reference_kernel.cl"
  #endif
#else
  #if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/mtran/mtran_kernel.cu"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/mtran/mtran_reference_kernel.cu"
  #else
    #define KTT_KERNEL_FILE "../../examples/mtran/mtran_kernel.cu"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/mtran/mtran_reference_kernel.cu"
  #endif
#endif

int main(int argc, char **argv)
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
#if USE_PROFILING == 0
    const int width = 8192;
    const int height = 8192;
#else
    const int width = 4096;
    const int height = 4096;
#endif
    const ktt::DimensionVector ndRangeDimensions(width, height);
    const ktt::DimensionVector ndRangeDimensionsReference(width/16, height/16);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare data variables
    std::vector<float> dst(width * height);
    std::vector<float> src(width * height);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);
    for (int i = 0; i < width*height; i++)
    {
        src[i] = distribution(engine);
    }

    // Create tuner
#if USE_CUDA == 0
    ktt::Tuner tuner(platformIndex, deviceIndex);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::CUDA);
#else
     ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);
  #if USE_PROFILING == 1
    printf("Executing with profiling switched ON.\n");
    tuner.setKernelProfiling(true);
  #endif
#endif
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    // Create kernel and configure input/output
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "mtran", ndRangeDimensions, ktt::DimensionVector(1, 1));
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "mtranReference", ndRangeDimensionsReference, referenceWorkGroupDimensions);
    ktt::ArgumentId srcId = tuner.addArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId dstId = tuner.addArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly);
    ktt::ArgumentId widthId = tuner.addArgumentScalar(width);
    ktt::ArgumentId heightId = tuner.addArgumentScalar(height);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{dstId, srcId, widthId, heightId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{dstId, srcId, widthId, heightId});

    // Create tuning space
    tuner.addParameter(kernelId, "LOCAL_MEM", { 0, 1 });
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4, 8 });
#else
    tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4 });
#endif
    tuner.addParameter(kernelId, "CR", { 0, 1 });
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "PREFETCH", { 0, 1, 2 });
#endif
    tuner.addParameter(kernelId, "PADD_LOCAL", { 0, 1 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "TILE_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "TILE_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
    
    // Constraint tuning space
    auto xConstraint = [] (std::vector<size_t> v) { return (v[0] == v[1]); };
    auto yConstraint = [] (std::vector<size_t> v) { return (v[1] <= v[0]); };
    auto tConstraint = [] (std::vector<size_t> v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
    auto pConstraint = [] (std::vector<size_t> v) { return (v[0] || !v[1]); };
    auto vConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] <= 64);  };
    auto vlConstraint = [] (std::vector<size_t> v) { return (!v[0] || v[1] == 1);  };
    auto minparConstraint = [] (std::vector<size_t> v) {return (v[0] * v[1] >= 32);};
    tuner.addConstraint(kernelId, { "TILE_SIZE_X", "WORK_GROUP_SIZE_X" }, xConstraint);
    tuner.addConstraint(kernelId, { "TILE_SIZE_Y", "WORK_GROUP_SIZE_Y" }, yConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" }, tConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "PADD_LOCAL" }, pConstraint);
    tuner.addConstraint(kernelId, { "TILE_SIZE_X", "VECTOR_TYPE" }, vConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "VECTOR_TYPE" }, vlConstraint);
//    tuner.addConstraint(kernelId, { "TILE_SIZE_X", "TILE_SIZE_Y" }, minparConstraint);

    // Configure parallelism
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
    auto xGlobalModifier = [](const size_t size, const std::vector<size_t>& vector) {return size / vector.at(0) / vector.at(1);};
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, std::vector<std::string>{ "TILE_SIZE_X", "VECTOR_TYPE" }, xGlobalModifier);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y", ktt::ModifierAction::Divide);

    auto wgSize = [](const std::vector<size_t>& v) {return v[0]*v[1] >= 32;};
    tuner.addConstraint(kernelId, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, wgSize);

    // Assign reference and set error check
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{dstId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);

    // Perform tuning
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "mtran_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
