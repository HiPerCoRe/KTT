#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

// Problem size
const int width = 256;
const int height = 128;
const int depth = 128;

// Half-filter and filter size - hfs > 1 not supported for Sliding plane kernel
const int hfs = 1;
const int fs = 2 * hfs + 1;

// New NVidia GPUs have max.workgroup size of 1024
// My Intel(R) HD Graphics Kabylake ULT GT2 has max of 512
const int maxWorkGroupSize = 1024;

// Local memory size in bytes
const int maxLocalMemorySize = 32768;

// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(const size_t a, const size_t b)
{
    return (a + b - 1) / b;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return (a / b) * b == a;
}

int main(int argc, char **argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Examples/Convolution3d/Convolution3d.cl";
    std::string referenceKernelFile = kernelPrefix + "../Examples/Convolution3d/Convolution3dReference.cl";

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

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 3.0f);

    std::vector<float> src((depth + 2 * hfs) * (height + 2 * hfs) * (width + 2 * hfs));
    std::vector<float> dest(depth * height * width, 0.0f);
    std::vector<float> coeff(fs * fs * fs);

    // Initialize source matrix padded by zeros
    for (int d = 0; d < depth + 2 * hfs; ++d)
    {
        for (int h = 0; h < height + 2 * hfs; ++h)
        {
            for (int w = 0; w < width + 2 * hfs; ++w)
            {
                const int index = d * (width + 2 * hfs) * (height + 2 * hfs) + h * (width + 2 * hfs) + w;

                if (d < hfs || d > depth - 1 + hfs || h < hfs || h > height - 1 + hfs || w < hfs || w > width - 1 + hfs)
                {
                    src[index] = 0.0f;
                }
                else
                {
                    src[index] = distribution(engine);
                }
            }
        }
    }

    // Creates the filter coefficients (gaussian blur)
    float sigma = 1.0f;
    float sum = 0.0f;

    for (int x = -hfs; x <= hfs; ++x)
    {
        for (int y = -hfs; y <= hfs; ++y)
        {
            for (int z = -hfs; z <= hfs; ++z)
            {
                const float exponent = -0.5f * (pow(x / sigma, 2.0f) + pow(y / sigma, 2.0f) + pow(z / sigma, 2.0f));
                const float c = static_cast<float>(exp(exponent) / (pow(2.0f * 3.14159265f, 1.5f) * pow(sigma, 3.0f)));
                sum += c;
                coeff[(z + hfs) * fs * fs + (y + hfs) * fs + (x + hfs)] = c;
            }
        }
    }

    for (auto &item : coeff)
    {
        item = item / sum;
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Kernel dimensions
    const ktt::DimensionVector ndRangeDimensions(width, height, depth);
    const ktt::DimensionVector workGroupDimensions;

    // Add 3 kernels to the tuner, one of them acts as reference kernel
    const ktt::KernelDefinitionId blockedDefinition = tuner.AddKernelDefinitionFromFile("conv", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId slidingPlaneDefinition = tuner.AddKernelDefinitionFromFile("conv2", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("conv_reference", referenceKernelFile,
        ndRangeDimensions, workGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateCompositeKernel("3D Convolution", {blockedDefinition, referenceDefinition,
        slidingPlaneDefinition}, [blockedDefinition, referenceDefinition, slidingPlaneDefinition](ktt::ComputeInterface& interface)
    {
        const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
        const uint64_t algorithm = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "ALGORITHM");

        if (algorithm == 0)
        {
            interface.RunKernel(referenceDefinition);
        }
        else if (algorithm == 1)
        {
            interface.RunKernel(blockedDefinition);
        }
        else
        {
            interface.RunKernel(slidingPlaneDefinition);
        }
    });

    // Add kernel parameters.
    // ALGORITHM 0 - Reference kernel, 1 - Blocked kernel, 2 - Sliding plane kernel
    tuner.AddParameter(kernel, "ALGORITHM", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "TBX", std::vector<uint64_t>{8, 16, 32, 64});
    tuner.AddParameter(kernel, "TBY", std::vector<uint64_t>{8, 16, 32, 64});
    tuner.AddParameter(kernel, "TBZ", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "LOCAL", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "WPTX", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "WPTY", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "WPTZ", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "VECTOR", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, fs});
    tuner.AddParameter(kernel, "CONSTANT_COEFF", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "CACHE_WORK_TO_REGS", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "REVERSE_LOOP_ORDER", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "REVERSE_LOOP_ORDER2", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "REVERSE_LOOP_ORDER3", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "PADDING", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{4, 8, 16, 32});

    // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
    // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
    // extra (halo) threads only to load the padding to local memory - they don't compute.
    std::vector<uint64_t> integers{1, 2, 3, 4, 8, 9, 10, 16, 17, 18, 32, 33, 34, 64, 65, 66};

    tuner.AddParameter(kernel, "TBX_XL", integers);
    tuner.AddParameter(kernel, "TBY_XL", integers);
    tuner.AddParameter(kernel, "TBZ_XL", integers);

    // Modify XY NDRange size for all kernels
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& v)
    {
        return (size * v[0] / (v[1] * v[2]));
    };

    tuner.AddThreadModifier(kernel, {blockedDefinition, slidingPlaneDefinition, referenceDefinition}, ktt::ModifierType::Global,
        ktt::ModifierDimension::X, {"TBX_XL", "TBX", "WPTX"}, globalModifier);
    tuner.AddThreadModifier(kernel, {blockedDefinition, slidingPlaneDefinition, referenceDefinition}, ktt::ModifierType::Global,
        ktt::ModifierDimension::Y, {"TBY_XL", "TBY", "WPTY"}, globalModifier);

    // Modify Z NDRange size for Blocked kernel
    tuner.AddThreadModifier(kernel, {blockedDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
        {"TBZ_XL", "TBZ", "WPTZ"}, globalModifier);

    // Modify Z NDRange size for Sliding plane kernel
    auto globalModifierZ = [](const uint64_t size, const std::vector<uint64_t>& v)
    {
        return (size * v[0] / (v[1] * v[2] * v[3]));
    };

    tuner.AddThreadModifier(kernel, {slidingPlaneDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
        {"TBZ_XL", "TBZ", "WPTZ", "Z_ITERATIONS"}, globalModifierZ);

    // Modify workgroup size for all kernels
    tuner.AddThreadModifier(kernel, {blockedDefinition, slidingPlaneDefinition, referenceDefinition}, ktt::ModifierType::Local,
        ktt::ModifierDimension::X, "TBX_XL", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {blockedDefinition, slidingPlaneDefinition, referenceDefinition}, ktt::ModifierType::Local,
        ktt::ModifierDimension::Y, "TBY_XL", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {blockedDefinition, slidingPlaneDefinition, referenceDefinition}, ktt::ModifierType::Local,
        ktt::ModifierDimension::Z, "TBZ_XL", ktt::ModifierAction::Multiply);

    // For LOCAL == 2, extend block size by halo threads
    auto HaloThreads = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 2)
        {
            return (v[1] == v[2] + CeilDiv(2 * hfs, v[3]));
        }
        else
        {
            return (v[1] == v[2]);
        }
    };

    tuner.AddConstraint(kernel, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, HaloThreads);
    tuner.AddConstraint(kernel, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, HaloThreads);
    tuner.AddConstraint(kernel, {"LOCAL", "TBZ_XL", "TBZ", "WPTZ"}, HaloThreads);

    // Sets padding to zero in case local memory is not used
    auto padding = [](const std::vector<uint64_t>& v) { return (v[0] != 0 || v[1] == 0); };
    tuner.AddConstraint(kernel, {"LOCAL", "PADDING"}, padding);

    // GPUs have max. workgroup size
    auto maxWgSize = [](const std::vector<uint64_t>& v)
    {
        return v[0] * v[1] * v[2] <= maxWorkGroupSize;
    };

    tuner.AddConstraint(kernel, {"TBX_XL", "TBY_XL", "TBZ_XL"}, maxWgSize);

    // GPUs have max. local memory size
    auto maxLocalMemSize = [](const std::vector<uint64_t>& v)
    {
        const uint64_t haloXY = v[1] == 1 ? 2 * hfs : 0;
        const uint64_t haloZ = v[0] == 2 || v[1] == 1 ? 2 * hfs : 0;
        return v[1] == 0 || (v[3] * v[4] + haloXY + v[2]) * (v[5] * v[6] + haloXY) * (v[7] * v[8] + haloZ)
            * sizeof(float) <= maxLocalMemorySize;
    };

    tuner.AddConstraint(kernel, {"ALGORITHM", "LOCAL", "PADDING", "TBX_XL", "WPTX", "TBY_XL", "WPTY", "TBZ_XL", "WPTZ"},
        maxLocalMemSize);

    auto reverseCacheLoopsOrder = [](const std::vector<uint64_t>& v) { return v[0] == 1 || v[1] == 0; };
    tuner.AddConstraint(kernel, {"CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER3"}, reverseCacheLoopsOrder);

    // Sets the constrains on the vector size
    auto vectorConstraint = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 2)
        {
            return IsMultiple(v[2], v[1]) && IsMultiple(2 * hfs, v[1]);
        }
        else
        {
            return IsMultiple(v[2], v[1]);
        }
    };

    tuner.AddConstraint(kernel, {"LOCAL", "VECTOR", "WPTX"}, vectorConstraint);

    auto algorithm = [](const std::vector<uint64_t>& v)
    {
        // Don't tune any parameters for the reference kernel (ALGORITHM == 0)
        if (v[0] == 0)
        {
            return v[1] == 8 && v[2] == 8 && v[3] == 1 && v[4] == 1 && v[5] == 1 && v[6] == 1 && v[7] == 0 && v[8] == 1
                && v[9] == 1 && v[10] == 1 && v[11] == 1 && v[12] == 1 && v[13] == 1 && v[14] == 1;
        }
        // Tune everything for Blocked kernel (ALGORITHM == 1)
        else if (v[0] == 1)
        {
            return true;
        }
        // Set TBZ to 1, WPTZ to 1, and LOCAL to 1/2 for Sliding plane kernel (ALGORITHM == 2)
        else // v[0] == 2
        {
            return (v[3] == 1 && v[6] == 1 && v[7] != 0);
        }
    };

    tuner.AddConstraint(kernel, {"ALGORITHM", "TBX", "TBY", "TBZ", "WPTX", "WPTY", "WPTZ", "LOCAL", "VECTOR", "UNROLL_FACTOR",
        "CONSTANT_COEFF", "CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER", "REVERSE_LOOP_ORDER2", "REVERSE_LOOP_ORDER3"}, algorithm);

    auto slidingPlane = [](const std::vector<uint64_t>& v) { return v[0] == 2 || v[1] == 16; };
    tuner.AddConstraint(kernel, {"ALGORITHM", "Z_ITERATIONS"}, slidingPlane);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId widthId = tuner.AddArgumentScalar(width);
    const ktt::ArgumentId heightId = tuner.AddArgumentScalar(height);
    const ktt::ArgumentId depthId = tuner.AddArgumentScalar(depth);
    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId coeffId = tuner.AddArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId destId = tuner.AddArgumentVector(dest, ktt::ArgumentAccessType::WriteOnly);

    // Set kernel arguments for both tuned kernel and reference kernel
    tuner.SetArguments(blockedDefinition, {widthId, heightId, srcId, coeffId, destId});
    tuner.SetArguments(referenceDefinition, {widthId, heightId, srcId, coeffId, destId});
    tuner.SetArguments(slidingPlaneDefinition, {widthId, heightId, depthId, srcId, coeffId, destId});

    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
    tuner.SetReferenceComputation(destId, [&src, &coeff](void* buffer)
    {
        float* output = static_cast<float*>(buffer);

        for (int d = 0; d < depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    float acc = 0.0f;

                    for (int k = -hfs; k <= hfs; ++k)
                    {
                        for (int l = -hfs; l <= hfs; ++l)
                        {
                            for (int m = -hfs; m <= hfs; ++m)
                            {
                                acc += coeff[(k + hfs) * fs * fs + (l + hfs) * fs + (m + hfs)]
                                    * src[(d + hfs + k) * (width + 2 * hfs) * (height + 2 * hfs)
                                    + (h + hfs + l) * (width + 2 * hfs) + (w + hfs + m)];
                            }
                        }
                    }

                    output[d * width * height + h * width + w] = acc;
                }
            }
        }
    });

    // Launch kernel tuning
    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "Convolution3dOutput", ktt::OutputFormat::XML);

    return 0;
};
