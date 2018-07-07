#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/cltune-gemm/gemm.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/cltune-gemm/gemm_reference.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/cltune-gemm/gemm.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/cltune-gemm/gemm_reference.cl"
#endif

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) {
    return ((a/b)*b == a) ? true : false;
};
class tunable: public ktt::TuningManipulator {
    public:

        tunable(uint32_t kSizeM, uint32_t kSizeN)
        {
            this->kSizeM = kSizeM;
            this->kSizeN = kSizeN;
        }

        void launchComputation(const ktt::KernelId kernelId) override {
            std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
            int mdimc = (int)parameterValues[3].getValue();
            int ndimc = (int)parameterValues[4].getValue();
            int mwg = (int)parameterValues[0].getValue();
            int nwg = (int)parameterValues[1].getValue();
            // Sets the constraints for local memory size limitations
            //       auto LocalMemorySize = [] (std::vector<size_t> v) {
            //         return (((v[0]*v[1]*v[2]/v[3]) + (v[4]*v[5]*v[6]/v[7]))*sizeof(float));
            //       };
            //tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG", "VWM", "SB", "KWG", "NWG", "VWN"});
            // Modifies the thread-sizes (both global and local) based on the parameters
            const ktt::DimensionVector ndRangeDimensions(kSizeM*mdimc/mwg, kSizeN*ndimc/nwg);
            const ktt::DimensionVector workGroupDimensions(mdimc, ndimc);
            runKernel(kernelId, ndRangeDimensions, workGroupDimensions);
        }
    private:
        uint32_t kSizeM;
        uint32_t kSizeN;
};

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

    // Declare data variables
    const uint32_t kSizeM = 2048;
    const uint32_t kSizeN = 2048;
    const uint32_t kSizeK = 2048;

    const ktt::DimensionVector ndRangeDimensions(kSizeM, kSizeN);
    const ktt::DimensionVector workGroupDimensions(1,1);
    const ktt::DimensionVector referenceWorkGroupDimensions(8,8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    auto mat_a = std::vector<float>(kSizeM*kSizeK);
    auto mat_b = std::vector<float>(kSizeN*kSizeK);
    auto mat_c = std::vector<float>(kSizeM*kSizeN);
    for (uint32_t i = 0; i < kSizeM*kSizeK; i++)
        mat_a.at(i) = distribution(engine);
    for (uint32_t i = 0; i < kSizeN*kSizeK; i++)
        mat_b.at(i) = distribution(engine);
    for (uint32_t i = 0; i < kSizeM*kSizeN; i++)
        mat_c.at(i) = 0.0f;

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "gemm_fast", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "gemm_reference", ndRangeDimensions, referenceWorkGroupDimensions);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, "MWG", {16, 32, 64, 128});
    tuner.addParameter(kernelId, "NWG", {16, 32, 64, 128});
    tuner.addParameter(kernelId, "KWG", {16, 32});
    tuner.addParameter(kernelId, "MDIMC", {8, 16, 32});
    tuner.addParameter(kernelId, "NDIMC", {8, 16, 32});
    tuner.addParameter(kernelId, "MDIMA", {8, 16, 32});
    tuner.addParameter(kernelId, "NDIMB", {8, 16, 32});
    tuner.addParameter(kernelId, "KWI", {2, 8});
    tuner.addParameter(kernelId, "VWM", {1, 2, 4, 8});
    tuner.addParameter(kernelId, "VWN", {1, 2, 4, 8});
    tuner.addParameter(kernelId, "STRM", {0, 1});
    tuner.addParameter(kernelId, "STRN", {0, 1});
    tuner.addParameter(kernelId, "SA", {0, 1});
    tuner.addParameter(kernelId, "SB", {0, 1});
    tuner.addParameter(kernelId, "PRECISION", {32});


    // Add all arguments utilized by kernels
    ktt::ArgumentId kSizeMId = tuner.addArgumentScalar(kSizeM);
    ktt::ArgumentId kSizeNId = tuner.addArgumentScalar(kSizeN);
    ktt::ArgumentId kSizeKId = tuner.addArgumentScalar(kSizeK);
    ktt::ArgumentId matAId = tuner.addArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId matBId = tuner.addArgumentVector(mat_b, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId matCId = tuner.addArgumentVector(std::vector<float>(mat_c), ktt::ArgumentAccessType::WriteOnly);

    // Add conditions
    // Sets constraints: Set-up the constraints functions to use. The constraints require a function
    // object (in this case a lambda) which takes a vector of tuning parameter values and returns
    // a boolean value whether or not the tuning configuration is legal. In this case, the helper
    // function 'IsMultiple' is employed for convenience. In the calls to 'AddConstraint' below, the
    // vector of parameter names (as strings) matches the input integer vector of the lambda's.
    auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
    auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
    auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };

    // Sets constraints: Requirement for unrolling the KWG loop
    tuner.addConstraint(kernelId, {"KWG", "KWI"}, MultipleOfX);

    // Sets constraints: Required for integer MWI and NWI
    tuner.addConstraint(kernelId, {"MWG", "MDIMC", "VWM"}, MultipleOfXMulY);
    tuner.addConstraint(kernelId, {"NWG", "NDIMC", "VWN"}, MultipleOfXMulY);

    // Sets constraints: Required for integer MWIA and NWIB
    tuner.addConstraint(kernelId, {"MWG", "MDIMA", "VWM"}, MultipleOfXMulY);
    tuner.addConstraint(kernelId, {"NWG", "NDIMB", "VWN"}, MultipleOfXMulY);

    // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
    tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, MultipleOfXMulYDivZ);
    tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, MultipleOfXMulYDivZ);


    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId}); 
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{kSizeMId, kSizeNId, kSizeKId, matAId, matBId, matCId}); 

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{matCId});

    tunable* t = new tunable(kSizeM, kSizeN);
    tuner.setTuningManipulator(kernelId, std::unique_ptr<tunable>(t));  

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "gemm_output.csv", ktt::PrintFormat::CSV);

    return 0;
};

