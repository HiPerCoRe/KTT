#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/mtran/mtran_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/mtran/mtran_reference_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/mtran/mtran_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/mtran/mtran_reference_kernel.cl"
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
    const int width = 8192;
    const int height = 8192;
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
    ktt::Tuner tuner(platformIndex, deviceIndex);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::CUDA);

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
    tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4, 8 });
    tuner.addParameter(kernelId, "CR", { 0, 1 });
    tuner.addParameter(kernelId, "PREFETCH", { 0, 1, 2 });
    tuner.addParameter(kernelId, "PADD_LOCAL", { 0, 1 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 16/*1, 2, 4, 8, 16, 32, 64*/ });
    tuner.addParameter(kernelId, "TILE_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "TILE_SIZE_Y", { 16/*1, 2, 4, 8, 16, 32, 64*/ });
    
    // Constraint tuning space
    auto xConstraint = [] (std::vector<size_t> v) { return (v[0] == v[1]); };
    auto yConstraint = [] (std::vector<size_t> v) { return (v[1] <= v[0]); };
    auto tConstraint = [] (std::vector<size_t> v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
    auto pConstraint = [] (std::vector<size_t> v) { return (v[0] || !v[1]); };
    auto vConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] <= 64);  };
    auto vlConstraint = [] (std::vector<size_t> v) { return (!v[0] || v[1] == 1);  };
    tuner.addConstraint(kernelId, { "TILE_SIZE_X", "WORK_GROUP_SIZE_X" }, xConstraint);
    tuner.addConstraint(kernelId, { "TILE_SIZE_Y", "WORK_GROUP_SIZE_Y" }, yConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" }, tConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "PADD_LOCAL" }, pConstraint);
    tuner.addConstraint(kernelId, { "TILE_SIZE_X", "VECTOR_TYPE" }, vConstraint);
    tuner.addConstraint(kernelId, { "LOCAL_MEM", "VECTOR_TYPE" }, vlConstraint);

    // Configure parallelism
    /*tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "TILE_SIZE_X", ktt::ModifierAction::Divide);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y", ktt::ModifierAction::Divide);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_TYPE", ktt::ModifierAction::Divide);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);*/
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "TILE_SIZE_X", ktt::ModifierAction::Divide);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y", ktt::ModifierAction::Divide);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_TYPE", ktt::ModifierAction::Divide);
//    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
//    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);

    // Assign reference and set error check
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{dstId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);

    // Perform tuning
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "mtran_output.csv", ktt::PrintFormat::CSV);

/*    cltune::Tuner tuner(platformIndex, deviceIndex);
    size_t kernelId = tuner.AddKernel(std::vector<std::string>{ TUNED_KERNEL_NAME }, "mtran", ndRangeDimensions, { 1, 1 });
    tuner.addParameter(kernelId, "ONE", { 1 });
    tuner.addParameter(kernelId, "LOCAL_MEM", { 0, 1 });
    tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4, 8 });
    //tuner.addParameter(kernelId, "VECTOR_TYPE", { 1 });
    tuner.addParameter(kernelId, "CR", { 0, 1 });
    tuner.addParameter(kernelId, "PREFETCH", { 0, 1, 2 });
    tuner.addParameter(kernelId, "PADD_LOCAL", { 0, 1 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "TILE_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
    tuner.addParameter(kernelId, "TILE_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
    auto xConstraint = [] (std::vector<size_t> v) { return (v[0] == v[1]); };
    auto yConstraint = [] (std::vector<size_t> v) { return (v[1] <= v[0]); };
    auto tConstraint = [] (std::vector<size_t> v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
    auto pConstraint = [] (std::vector<size_t> v) { return (v[0] || !v[1]); };
    auto vConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] <= 64);  };
    auto vlConstraint = [] (std::vector<size_t> v) { return (!v[0] || v[1] == 1);  };
    tuner.addConstraint(kernelId, xConstraint, { "TILE_SIZE_X", "WORK_GROUP_SIZE_X" });
    tuner.addConstraint(kernelId, yConstraint, { "TILE_SIZE_Y", "WORK_GROUP_SIZE_Y" });
    tuner.addConstraint(kernelId, tConstraint, { "LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" });
    tuner.addConstraint(kernelId, pConstraint, { "LOCAL_MEM", "PADD_LOCAL" } );
    tuner.addConstraint(kernelId, vConstraint, { "TILE_SIZE_X", "VECTOR_TYPE" });
    tuner.addConstraint(kernelId, vlConstraint, { "LOCAL_MEM", "VECTOR_TYPE" } );
    tuner.MulLocalSize(kernelId, { "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" });
    tuner.DivGlobalSize(kernelId, { "TILE_SIZE_X", "TILE_SIZE_Y" });
    tuner.DivGlobalSize(kernelId, { "VECTOR_TYPE", "ONE" });
    tuner.MulGlobalSize(kernelId, { "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" });

    tuner.SetReference(std::vector<std::string>{ REFERENCE_KERNEL_NAME }, "mtranReference", ndRangeDimensions, { 16, 16 });

    tuner.AddArgumentOutput(dst);
    tuner.AddArgumentInput(src);
    tuner.AddArgumentScalar(width);
    tuner.AddArgumentScalar(height);

    tuner.Tune();
    tuner.PrintToScreen();
    tuner.PrintToFile("result.csv");*/

    return 0;
}
