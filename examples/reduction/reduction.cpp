#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"
#include "reduction_reference.h"
#include "reduction_tunable.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/reduction/reduction_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/reduction/reduction_kernel.cl"
#endif

int main(int argc, char** argv)
{
    // Initialize platform and device index
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
            }
        }
    }

    // Declare and initialize data
    const int n = 32*1024*1024;
    const int nAlloc = ((n+16-1)/16)*16; // padd to longest vector size
    std::vector<float> src(nAlloc, 0.0f);
    std::vector<float> dst(nAlloc, 0.0f);

    for (int i = 0; i < n; i++)
    {
        src[i] = 1000.0f*((float)rand()) / ((float) RAND_MAX);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex);

    // create kernel
    int nUp = ((n+512-1)/512)*512; // maximum WG size used in tuning parameters
    ktt::DimensionVector ndRangeDimensions(nUp);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "reduce", ndRangeDimensions, workGroupDimensions);

    // create input/output
    ktt::ArgumentId srcId = tuner.addArgumentVector(src, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId dstId = tuner.addArgumentVector(dst, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId nId = tuner.addArgumentScalar(n);
    int offset = 0;
    ktt::ArgumentId inOffsetId = tuner.addArgumentScalar(offset);
    ktt::ArgumentId outOffsetId = tuner.addArgumentScalar(offset);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{srcId, dstId, nId, inOffsetId, outOffsetId});

    // get number of compute units
    const ktt::DeviceInfo di = tuner.getCurrentDeviceInfo();
    std::cout << "Number of compute units: " << di.getMaxComputeUnits() << std::endl;
    size_t cus = di.getMaxComputeUnits();

    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 128, 256, 512},
        ktt::ModifierType::Local,
        ktt::ModifierAction::Multiply,
        ktt::ModifierDimension::X);
    tuner.addParameter(kernelId, "UNBOUNDED_WG", {0, 1});
    tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
    tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2, 4, 8, 16},
        ktt::ModifierType::Global,
        ktt::ModifierAction::Divide,
        ktt::ModifierDimension::X);
    tuner.addParameter(kernelId, "USE_ATOMICS", {0, 1});

    auto persistConstraint = [](std::vector<size_t> v) {return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0);};
    tuner.addConstraint(kernelId, persistConstraint, {"UNBOUNDED_WG", "WG_NUM"});
    auto persistentAtomic = [](std::vector<size_t> v) {return (v[0] == 1) || (v[0] == 0 && v[1] == 1);};
    tuner.addConstraint(kernelId, persistentAtomic, {"UNBOUNDED_WG", "USE_ATOMICS"});
    auto unboundedWG = [](std::vector<size_t> v) {return (!v[0] || v[1] >= 32);};
    tuner.addConstraint(kernelId, unboundedWG, {"UNBOUNDED_WG", "WORK_GROUP_SIZE_X"});

    tuner.setReferenceClass(kernelId, std::make_unique<referenceReduction>(src, dstId), std::vector<ktt::ArgumentId>{dstId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, (float)n*500.0f/10'000'000.0f);
    tuner.setValidationRange(dstId, 1);

    tuner.setTuningManipulator(kernelId, std::make_unique<TunableReduction>(nAlloc, &src, &dst, srcId, dstId, nId, inOffsetId, outOffsetId));
    
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "reduction_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
