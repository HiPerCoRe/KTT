#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/cltune-conv/conv.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/cltune-conv/conv_reference.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/cltune-conv/conv.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/cltune-conv/conv_reference.cl"
#endif

// Settings (synchronise these with "conv.cpp", "conv.cl" and "conv_reference.cl")
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(const size_t a, const size_t b) {
    return (a + b - 1) / b;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b) {
    return ((a / b) * b == a) ? true : false;
}

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

    // Declare data variables
    const uint32_t kSizeX = 8192; // Matrix dimension X
    const uint32_t kSizeY = 4096; // Matrix dimension Y

    const ktt::DimensionVector ndRangeDimensions(kSizeX, kSizeY);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    std::vector<float> mat_a((kSizeX + 2 * HFS) * (kSizeY + 2 * HFS), 0.0f);
    std::vector<float> mat_b(kSizeX * kSizeY, 0.0f);
    std::vector<float> coeff(FS * FS);

    // Populates input data structure by padded data
    for (size_t i = 0; i < kSizeY; i++)
        for (size_t j = 0; j < kSizeX; j++)
            mat_a[(i + HFS) * (kSizeX + 2 * HFS) + j + HFS] = distribution(engine);

    // Creates the filter coefficients (gaussian blur)
    float sigma = 1.0f;
    float mean = FS / 2.0f;
    float sum = 0.0f;
    for (size_t x = 0; x < FS; ++x) {
        for (size_t y = 0; y < FS; ++y) {
            float exponent = -0.5f * (pow((x - mean) / sigma, 2.0f) + pow((y - mean) / sigma, 2.0f));
            coeff[y * FS + x] = static_cast<float>(exp(exponent) / (2.0f * 3.14159265f * sigma * sigma));
            sum += coeff[y * FS + x];
        }
    }
    for (auto& item : coeff) {item = item / sum;}

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "conv", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "conv_reference", ndRangeDimensions, referenceWorkGroupDimensions);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, "TBX", {8, 16, 32, 64});
    tuner.addParameter(kernelId, "TBY", {8, 16, 32, 64});
    tuner.addParameter(kernelId, "LOCAL", {0, 1, 2});
    tuner.addParameter(kernelId, "WPTX", {1, 2, 4, 8});
    tuner.addParameter(kernelId, "WPTY", {1, 2, 4, 8});
    tuner.addParameter(kernelId, "VECTOR", {1, 2, 4});
    tuner.addParameter(kernelId, "UNROLL_FACTOR", {1, FS});
    tuner.addParameter(kernelId, "PADDING", {0, 1});

    // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
    // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
    // extra threads to compute the halo threads. How many extra threads are needed is dependend on
    // the filter size. Here we support a the TBX and TBY size plus up to 10 extra threads.
    std::vector<size_t> integers{8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};

    tuner.addParameter(kernelId, "TBX_XL", integers);
    tuner.addParameter(kernelId, "TBY_XL", integers);

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const size_t size, const std::vector<size_t>& vector) {return (size * vector.at(0)) / (vector.at(1) * vector.at(2));};
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, std::vector<std::string>{"TBX_XL", "TBX", "WPTX"},
        globalModifier);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, std::vector<std::string>{"TBY_XL", "TBY", "WPTY"},
        globalModifier);

    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "TBX_XL", ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "TBY_XL", ktt::ModifierAction::Multiply);

    // Add constraints
    auto HaloThreads = [](const std::vector<size_t>& v) {
        if (v[0] == 2) {return (v[1] == v[2] + CeilDiv(2 * HFS, v[3]));} // With halo threads
        else           {return (v[1] == v[2]);}                       // Without halo threads
    };
    tuner.addConstraint(kernelId, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, HaloThreads);
    tuner.addConstraint(kernelId, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, HaloThreads);

    // Sets the constrains on the vector size
    auto VectorConstraint = [](const std::vector<size_t>& v) {
        if (v[0] == 2) {return IsMultiple(v[2], v[1]) && IsMultiple(2 * HFS, v[1]);}
        else           {return IsMultiple(v[2], v[1]);}
    };
    tuner.addConstraint(kernelId, {"LOCAL", "VECTOR", "WPTX"}, VectorConstraint);

    // Makes sure the work per thread is not too high, otherwise too many registers would be used.
    //auto WorkPerThreadConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] < 32); };
    //tuner.AddConstraint(id, WorkPerThreadConstraint, {"WPTX", "WPTY"});

    // Sets padding to zero in case local memory is not used
    auto PaddingConstraint = [](const std::vector<size_t>& v) {return (v[1] == 0 || v[0] != 0);};
    tuner.addConstraint(kernelId, {"LOCAL", "PADDING"}, PaddingConstraint);

    // Add all arguments utilized by kernels
    ktt::ArgumentId kSizeXId = tuner.addArgumentScalar(kSizeX);
    ktt::ArgumentId kSizeYId = tuner.addArgumentScalar(kSizeY);
    ktt::ArgumentId matAId = tuner.addArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId coeffId = tuner.addArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId matBId = tuner.addArgumentVector(mat_b, ktt::ArgumentAccessType::WriteOnly);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{kSizeXId, kSizeYId, matAId, coeffId, matBId}); 
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{kSizeXId, kSizeYId, matAId, coeffId, matBId}); 

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{matBId});

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "conv_output.csv", ktt::PrintFormat::CSV);

    return 0;
};
