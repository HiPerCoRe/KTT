#include "../ExampleReferenceComputation.h"

using namespace std;

class Convolution3d: public ExampleReferenceComputation 
{
protected:
    Convolution3d(
        int argc,
        char** argv,
        int defaultProblemSize,
        string exampleFolderPath, string defaultKernelFileBaseName,
        bool rapidTest,
        bool useProfiling
    ):
    ExampleReferenceComputation(argc, argv, defaultProblemSize,
        exampleFolderPath, defaultKernelFileBaseName, rapidTest, useProfiling
    )
    {
        // cbrt is cube root, so that problem size N roughly translates to N MiB of values
        m_width = cbrt(m_problemSize)*256;
        m_height = cbrt(m_problemSize)*128;
        m_depth = cbrt(m_problemSize)*128;
    }

    friend ExampleReferenceComputation;

    int m_width, m_height, m_depth;
    ktt::KernelDefinitionId m_blockedDefinition, m_slidingPlaneDefinition;
    vector<float> m_src, m_dest, m_coeff;
    ktt::ArgumentId m_destId;

    // Half-filter and filter size - m_hfs > 1 not supported for Sliding plane kernel
    const int m_hfs = 1;
    const int m_fs = 2 * m_hfs + 1;

    // New NVidia GPUs have max.workgroup size of 1024
    // My Intel(R) HD Graphics Kabylake ULT GT2 has max of 512
    const unsigned m_maxWorkGroupSize = 1024;

    // Local memory size in bytes
    const unsigned m_maxLocalMemorySize = 32768;

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

    virtual void InitData() 
    {
        // Initialize data
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, 3.0f);

        m_src.resize((m_depth + 2 * m_hfs) * (m_height + 2 * m_hfs) * (m_width + 2 * m_hfs));
        m_dest.resize(m_depth * m_height * m_width, 0.0f);
        m_coeff.resize(m_fs * m_fs * m_fs);

        // Initialize source matrix padded by zeros
        for (int d = 0; d < m_depth + 2 * m_hfs; ++d)
        {
            for (int h = 0; h < m_height + 2 * m_hfs; ++h)
            {
                for (int w = 0; w < m_width + 2 * m_hfs; ++w)
                {
                    const int index = d * (m_width + 2 * m_hfs) * (m_height + 2 * m_hfs) + h * (m_width + 2 * m_hfs) + w;

                    if (d < m_hfs || d > m_depth - 1 + m_hfs || h < m_hfs || h > m_height - 1 + m_hfs || w < m_hfs || w > m_width - 1 + m_hfs)
                    {
                        m_src[index] = 0.0f;
                    }
                    else
                    {
                        m_src[index] = distribution(engine);
                    }
                }
            }
        }

        // Creates the filter coefficients (gaussian blur)
        float sigma = 1.0f;
        float sum = 0.0f;

        for (int x = -m_hfs; x <= m_hfs; ++x)
        {
            for (int y = -m_hfs; y <= m_hfs; ++y)
            {
                for (int z = -m_hfs; z <= m_hfs; ++z)
                {
                    const float exponent = -0.5f * (pow(x / sigma, 2.0f) + pow(y / sigma, 2.0f) + pow(z / sigma, 2.0f));
                    const float c = static_cast<float>(exp(exponent) / (pow(2.0f * 3.14159265f, 1.5f) * pow(sigma, 3.0f)));
                    sum += c;
                    m_coeff[(z + m_hfs) * m_fs * m_fs + (y + m_hfs) * m_fs + (x + m_hfs)] = c;
                }
            }
        }

        for (auto &item : m_coeff)
        {
            item = item / sum;
        }
    }
    virtual void InitKernels() 
    {
        // kernel dimensions
        const ktt::DimensionVector ndRangeDimensions(m_width, m_height, m_depth);
        const ktt::DimensionVector workGroupDimensions;

        // Add 3 kernels to the m_tuner, one of them acts as reference kernel
        m_blockedDefinition = m_tuner.AddKernelDefinitionFromFile("conv", m_kernelFile, ndRangeDimensions,
            workGroupDimensions);
        m_slidingPlaneDefinition = m_tuner.AddKernelDefinitionFromFile("conv2", m_kernelFile, ndRangeDimensions,
            workGroupDimensions);

        m_kernel = m_tuner.CreateCompositeKernel("3D Convolution", {m_blockedDefinition, m_slidingPlaneDefinition},
            [this](ktt::ComputeInterface& interface)
        {
            const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
            const uint64_t algorithm = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "ALGORITHM");

            if (algorithm == 0)
            {
                interface.RunKernel(m_blockedDefinition);
            }
            else
            {
                interface.RunKernel(m_slidingPlaneDefinition);
            }
        });

        // Add all arguments utilized by kernels
        const ktt::ArgumentId widthId = m_tuner.AddArgumentScalar(m_width);
        const ktt::ArgumentId heightId = m_tuner.AddArgumentScalar(m_height);
        const ktt::ArgumentId depthId = m_tuner.AddArgumentScalar(m_depth);
        const ktt::ArgumentId srcId = m_tuner.AddArgumentVector(m_src, ktt::ArgumentAccessType::ReadOnly);
        const ktt::ArgumentId coeffId = m_tuner.AddArgumentVector(m_coeff, ktt::ArgumentAccessType::ReadOnly);
        m_destId = m_tuner.AddArgumentVector(m_dest, ktt::ArgumentAccessType::WriteOnly);

        // Set kernel arguments for both tuned kernel and reference kernel
        m_tuner.SetArguments(m_blockedDefinition, {widthId, heightId, srcId, coeffId, m_destId});
        m_tuner.SetArguments(m_slidingPlaneDefinition, {widthId, heightId, depthId, srcId, coeffId, m_destId});
    }
    virtual void InitTuningParameters() 
    {
        // Add kernel parameters.
        // 0 - Blocked kernel, 1 - Sliding plane kernel
        m_tuner.AddParameter(m_kernel, "ALGORITHM", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "TBX", std::vector<uint64_t>{8, 16, 32, 64});
        m_tuner.AddParameter(m_kernel, "TBY", std::vector<uint64_t>{8, 16, 32, 64});
        m_tuner.AddParameter(m_kernel, "TBZ", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
        m_tuner.AddParameter(m_kernel, "LOCAL", std::vector<uint64_t>{0, 1, 2});
        m_tuner.AddParameter(m_kernel, "WPTX", std::vector<uint64_t>{1, 2, 4, 8});
        m_tuner.AddParameter(m_kernel, "WPTY", std::vector<uint64_t>{1, 2, 4, 8});
        m_tuner.AddParameter(m_kernel, "WPTZ", std::vector<uint64_t>{1, 2, 4, 8});
        m_tuner.AddParameter(m_kernel, "VECTOR", std::vector<uint64_t>{1, 2, 4});
        m_tuner.AddParameter(m_kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, static_cast<uint64_t>(m_fs)});
        m_tuner.AddParameter(m_kernel, "CONSTANT_COEFF", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "CACHE_WORK_TO_REGS", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "REVERSE_LOOP_ORDER", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "REVERSE_LOOP_ORDER2", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "REVERSE_LOOP_ORDER3", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "PADDING", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "Z_ITERATIONS", std::vector<uint64_t>{4, 8, 16, 32});

        // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
        // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
        // extra (halo) threads only to load the padding to local memory - they don't compute.
        std::vector<uint64_t> integers{1, 2, 3, 4, 8, 9, 10, 16, 17, 18, 32, 33, 34, 64, 65, 66};

        m_tuner.AddParameter(m_kernel, "TBX_XL", integers);
        m_tuner.AddParameter(m_kernel, "TBY_XL", integers);
        m_tuner.AddParameter(m_kernel, "TBZ_XL", integers);

        // Modify XY NDRange size for all kernels
        auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& v)
        {
            return (size / (v[0] * v[1]));
        };

        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition, m_slidingPlaneDefinition}, ktt::ModifierType::Global,
            ktt::ModifierDimension::X, {"TBX", "WPTX"}, globalModifier);
        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition, m_slidingPlaneDefinition}, ktt::ModifierType::Global,
            ktt::ModifierDimension::Y, {"TBY", "WPTY"}, globalModifier);

        // Modify Z NDRange size for Blocked kernel
        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"TBZ", "WPTZ"}, globalModifier);

        // Modify Z NDRange size for Sliding plane kernel
        auto globalModifierZ = [](const uint64_t size, const std::vector<uint64_t>& v)
        {
            return (size / (v[0] * v[1] * v[2]));
        };

        m_tuner.AddThreadModifier(m_kernel, {m_slidingPlaneDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"TBZ", "WPTZ", "Z_ITERATIONS"}, globalModifierZ);

        // Modify workgroup size for all kernels
        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition, m_slidingPlaneDefinition}, ktt::ModifierType::Local,
            ktt::ModifierDimension::X, "TBX_XL", ktt::ModifierAction::Multiply);
        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition, m_slidingPlaneDefinition}, ktt::ModifierType::Local,
            ktt::ModifierDimension::Y, "TBY_XL", ktt::ModifierAction::Multiply);
        m_tuner.AddThreadModifier(m_kernel, {m_blockedDefinition, m_slidingPlaneDefinition}, ktt::ModifierType::Local,
            ktt::ModifierDimension::Z, "TBZ_XL", ktt::ModifierAction::Multiply);

        // For LOCAL == 2, extend block size by halo threads
        auto HaloThreads = [this](const std::vector<uint64_t>& v)
        {
            if (v[0] == 2)
            {
                return (v[1] == v[2] + CeilDiv(2 * m_hfs, v[3]));
            }
            else
            {
                return (v[1] == v[2]);
            }
        };

        m_tuner.AddConstraint(m_kernel, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, HaloThreads);
        m_tuner.AddConstraint(m_kernel, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, HaloThreads);
        m_tuner.AddConstraint(m_kernel, {"LOCAL", "TBZ_XL", "TBZ", "WPTZ"}, HaloThreads);

        // Sets padding to zero in case local memory is not used
        auto padding = [](const std::vector<uint64_t>& v) { return (v[0] != 0 || v[1] == 0); };
        m_tuner.AddConstraint(m_kernel, {"LOCAL", "PADDING"}, padding);

        // GPUs have max. workgroup size
        auto maxWgSize = [this](const std::vector<uint64_t>& v)
        {
            return v[0] * v[1] * v[2] <= m_maxWorkGroupSize;
        };

        m_tuner.AddConstraint(m_kernel, {"TBX_XL", "TBY_XL", "TBZ_XL"}, maxWgSize);

        // GPUs have max. local memory size
        auto maxLocalMemSize = [this](const std::vector<uint64_t>& v)
        {
            const uint64_t haloXY = v[1] == 1 ? 2 * m_hfs : 0;
            const uint64_t haloZ = v[0] == 1 || v[1] == 1 ? 2 * m_hfs : 0;
            return v[1] == 0 || (v[3] * v[4] + haloXY + v[2]) * (v[5] * v[6] + haloXY) * (v[7] * v[8] + haloZ)
                * sizeof(float) <= m_maxLocalMemorySize;
        };

        m_tuner.AddConstraint(m_kernel, {"ALGORITHM", "LOCAL", "PADDING", "TBX_XL", "WPTX", "TBY_XL", "WPTY", "TBZ_XL", "WPTZ"},
            maxLocalMemSize);

        auto reverseCacheLoopsOrder = [](const std::vector<uint64_t>& v) { return v[0] == 1 || v[1] == 0; };
        m_tuner.AddConstraint(m_kernel, {"CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER3"}, reverseCacheLoopsOrder);

        // Sets the constrains on the vector size
        auto vectorConstraint = [this](const std::vector<uint64_t>& v)
        {
            if (v[0] == 2)
            {
                return IsMultiple(v[2], v[1]) && IsMultiple(2 * m_hfs, v[1]);
            }
            else
            {
                return IsMultiple(v[2], v[1]);
            }
        };

        m_tuner.AddConstraint(m_kernel, {"LOCAL", "VECTOR", "WPTX"}, vectorConstraint);

        auto algorithm = [](const std::vector<uint64_t>& v)
        {
            // Tune everything for Blocked kernel (ALGORITHM == 0)
            if (v[0] == 0)
            {
                return true;
            }
            // Set TBZ to 1, WPTZ to 1, and LOCAL to 1/2 for Sliding plane kernel (ALGORITHM == 1)
            else // v[0] == 1
            {
                return (v[3] == 1 && v[6] == 1 && v[7] != 0);
            }
        };

        m_tuner.AddConstraint(m_kernel, {"ALGORITHM", "TBX", "TBY", "TBZ", "WPTX", "WPTY", "WPTZ", "LOCAL", "VECTOR", "UNROLL_FACTOR",
            "CONSTANT_COEFF", "CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER", "REVERSE_LOOP_ORDER2", "REVERSE_LOOP_ORDER3"}, algorithm);

        auto slidingPlane = [](const std::vector<uint64_t>& v) { return v[0] == 1 || v[1] == 16; };
        m_tuner.AddConstraint(m_kernel, {"ALGORITHM", "Z_ITERATIONS"}, slidingPlane);
    }
    virtual void InitReference() 
    {
        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
        m_tuner.SetReferenceComputation(m_destId, [this](void* buffer)
        {
            float* output = static_cast<float*>(buffer);

            for (int d = 0; d < m_depth; ++d)
            {
                for (int h = 0; h < m_height; ++h)
                {
                    for (int w = 0; w < m_width; ++w)
                    {
                        float acc = 0.0f;

                        for (int k = -m_hfs; k <= m_hfs; ++k)
                        {
                            for (int l = -m_hfs; l <= m_hfs; ++l)
                            {
                                for (int m = -m_hfs; m <= m_hfs; ++m)
                                {
                                    acc += m_coeff[(k + m_hfs) * m_fs * m_fs + (l + m_hfs) * m_fs + (m + m_hfs)]
                                        * m_src[(d + m_hfs + k) * (m_width + 2 * m_hfs) * (m_height + 2 * m_hfs)
                                        + (h + m_hfs + l) * (m_width + 2 * m_hfs) + (w + m_hfs + m)];
                                }
                            }
                        }

                        output[d * m_width * m_height + h * m_width + w] = acc;
                    }
                }
            }
        });
    }
};


int main(int argc, char **argv)
{
    shared_ptr<Convolution3d> convolution3d = Convolution3d::Create<Convolution3d>(argc, argv, 1, "Examples/Convolution3d",
        "Convolution3d");

    // Launch kernel tuning
    convolution3d->Run();

    return 0;
};
