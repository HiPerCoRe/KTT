#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle kernel profiling.
const bool useProfiling = false;

std::vector<uint64_t> ParameterRange(const uint64_t max)
{
    std::vector<uint64_t> values;

    for (uint64_t i = 1; i <= max; ++i)
    {
        values.push_back(i);
    }

    return values;
}

enum class AtfSampleType
{
    Convolution,
    GEMM,
    CCSD,
    PRL
};

constexpr AtfSampleType activeSample = AtfSampleType::Convolution;
const std::string kernelPath = kernelPrefix + "../Examples/AtfSamples/";

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
        }
    }

    uint64_t inputSize1;
    uint64_t inputSize2;
    uint64_t inputSize3;
    uint64_t inputSize4;
    uint64_t inputSize5;
    uint64_t inputSize6;
    uint64_t inputSize7;

    if constexpr (activeSample == AtfSampleType::Convolution)
    {
        inputSize1 = 4096;
        inputSize2 = 4096;
    }
    else if constexpr (activeSample == AtfSampleType::GEMM)
    {
        inputSize1 = 10;
        inputSize2 = 500;
        inputSize3 = 64;
    }
    else if constexpr (activeSample == AtfSampleType::CCSD)
    {
        inputSize1 = 24;
        inputSize2 = 16;
        inputSize3 = 16;
        inputSize4 = 24;
        inputSize5 = 16;
        inputSize6 = 16;
        inputSize7 = 24;
    }
    else // AtfSampleType::PRL
    {
        inputSize1 = 1024;
        inputSize2 = 1024;
    }

    auto DescendingConstraint = [](const std::vector<uint64_t>& v)
    {
        bool valid = true;

        for (size_t i = 1; i < v.size(); ++i)
        {
            valid = valid && (v[i - 1] >= v[i]);
        }

        return valid;
    };

    auto UnequalConstraint = [](const std::vector<uint64_t>& v)
    {
        if (v.size() < 2)
        {
            return true;
        }

        bool valid = true;

        for (size_t i = 1; i < v.size(); ++i)
        {
            valid = valid && (v[i - 1] != v[i]);
        }

        valid = valid && (v[v.size() - 1] != v[0]);
        return valid;
    };

    auto LessThanOrEqualCeilDivConstraint = [](const std::vector<uint64_t>& v) { return v[0] <= (v[1] + v[2] - 1) / v[2]; };
    auto DividesConstraint = [](const std::vector<uint64_t>& v) { return v[1] % v[0] == 0; };
    auto DividesDivConstraint = [](const std::vector<uint64_t>& v) { return (v[1] / v[2]) % v[0] == 0; };
    auto NoPostInSecondKernelConstraint = [](const std::vector<uint64_t>& v) { return v[0] == 1 || (v[0] % v[1] == 0); };

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    if constexpr (computeApi == ktt::ComputeApi::CUDA && useProfiling) {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }
    ktt::KernelDefinitionId definition;
    ktt::KernelDefinitionId definition2;
    ktt::KernelId kernel;

    if constexpr (activeSample == AtfSampleType::Convolution)
    {
#if KTT_CUDA_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("gaussian_1", kernelPath + "GaussianStatic1.cu", ktt::DimensionVector(), ktt::DimensionVector());
#elif KTT_OPENCL_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("gaussian_1", kernelPath + "GaussianStatic1.cl", ktt::DimensionVector(), ktt::DimensionVector());
#endif
        kernel = tuner.CreateSimpleKernel("Convolution", definition);

        std::vector<float> in(inputSize1 * inputSize2);
        std::vector<float> out((inputSize1 - 4) * (inputSize2 - 4));
        std::vector<float> intRes((inputSize1 - 4) * (inputSize2 - 4));

        for (size_t i = 0; i < in.size(); ++i)
        {
            in[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = 0.0f;
        }

        for (size_t i = 0; i < intRes.size(); ++i)
        {
            intRes[i] = 0.0f;
        }

        const auto inId = tuner.AddArgumentVector(in, ktt::ArgumentAccessType::ReadOnly);
        const auto outId = tuner.AddArgumentVector(out, ktt::ArgumentAccessType::ReadWrite);
        const auto intResId = tuner.AddArgumentVector(intRes, ktt::ArgumentAccessType::ReadWrite);
        tuner.SetArguments(definition, {inId, outId, intResId});

        tuner.AddParameter(kernel, "CACHE_L_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "CACHE_P_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "G_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "L_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});
        tuner.AddParameter(kernel, "P_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});

        tuner.AddParameter(kernel, "OCL_DIM_L_1", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "OCL_DIM_L_2", std::vector<uint64_t>{0, 1});

        tuner.AddParameter(kernel, "INPUT_SIZE_L_1", std::vector<uint64_t>{inputSize1 - 4});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_1", ParameterRange(inputSize1 - 4));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_1", ParameterRange(inputSize1 - 4));
        tuner.AddParameter(kernel, "NUM_WG_L_1", ParameterRange(inputSize1 - 4));
        tuner.AddParameter(kernel, "NUM_WI_L_1", ParameterRange(inputSize1 - 4));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_2", std::vector<uint64_t>{inputSize2 - 4});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_2", ParameterRange(inputSize2 - 4));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_2", ParameterRange(inputSize2 - 4));
        tuner.AddParameter(kernel, "NUM_WG_L_2", ParameterRange(inputSize2 - 4));
        tuner.AddParameter(kernel, "NUM_WI_L_2", ParameterRange(inputSize2 - 4));

        tuner.AddParameter(kernel, "L_REDUCTION", std::vector<uint64_t>{1});
        tuner.AddParameter(kernel, "P_WRITE_BACK", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "L_WRITE_BACK", std::vector<uint64_t>{2});

        tuner.AddConstraint(kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_2"}, UnequalConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_2", "INPUT_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_2", "INPUT_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "INPUT_SIZE_L_2", "NUM_WG_L_2"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2] + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2] + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] + static_cast<uint64_t>(values[2] == 0) * values[3];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] + static_cast<uint64_t>(values[2] == 1) * values[3];
        });
    }
    else if constexpr (activeSample == AtfSampleType::GEMM)
    {
#if KTT_CUDA_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("gemm_1", kernelPath + "Gemm1.cu", ktt::DimensionVector(), ktt::DimensionVector());
        definition2 = tuner.AddKernelDefinitionFromFile("gemm_2", kernelPath + "Gemm2.cu", ktt::DimensionVector(), ktt::DimensionVector());
#elif KTT_OPENCL_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("gemm_1", kernelPath + "Gemm1.cl", ktt::DimensionVector(), ktt::DimensionVector());
        definition2 = tuner.AddKernelDefinitionFromFile("gemm_2", kernelPath + "Gemm2.cl", ktt::DimensionVector(), ktt::DimensionVector());
#endif

        std::vector<float> a(inputSize1 * inputSize3);
        std::vector<float> b(inputSize3 * inputSize2);
        std::vector<float> c(inputSize1 * inputSize2);
        std::vector<float> intRes(inputSize1 * inputSize2);
        std::vector<float> res(inputSize1 * inputSize2);
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            a[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < b.size(); ++i)
        {
            b[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < c.size(); ++i)
        {
            c[i] = 0.0f;
            intRes[i] = 0.0f;
            res[i] = 0.0f;
        }

        const size_t resSize = res.size() * sizeof(float);
        const auto aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
        const auto bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
        const auto cId = tuner.AddArgumentVector(c, ktt::ArgumentAccessType::ReadWrite);
        const auto resId = tuner.AddArgumentVector(res, ktt::ArgumentAccessType::ReadWrite);
        const auto intResId = tuner.AddArgumentVector(intRes, ktt::ArgumentAccessType::ReadWrite);
        tuner.SetArguments(definition, {aId, bId, resId, intResId});
        tuner.SetArguments(definition2, {intResId, resId, cId});

        kernel = tuner.CreateCompositeKernel("GEMM", {definition, definition2}, [resSize, resId, intResId, definition, definition2]
            (ktt::ComputeInterface& interface)
        {
            const auto& pairs = interface.GetCurrentConfiguration().GetPairs();
            size_t newResSize = resSize;

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "G_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            }

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1");
            }

            interface.ResizeBuffer(resId, newResSize, false);

            const size_t newIntResSize = resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            interface.ResizeBuffer(intResId, newIntResSize, false);

            interface.RunKernel(definition);

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1") > 1)
            {
                if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
                {
                    interface.ResizeBuffer(resId, resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1"), false);
                }

                interface.RunKernel(definition2);
            }
        });

        tuner.AddParameter(kernel, "CACHE_L_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "CACHE_P_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "G_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "L_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});
        tuner.AddParameter(kernel, "P_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});

        tuner.AddParameter(kernel, "OCL_DIM_L_1", std::vector<uint64_t>{0, 1, 2});
        tuner.AddParameter(kernel, "OCL_DIM_L_2", std::vector<uint64_t>{0, 1, 2});
        tuner.AddParameter(kernel, "OCL_DIM_R_1", std::vector<uint64_t>{0, 1, 2});

        tuner.AddParameter(kernel, "INPUT_SIZE_L_1", std::vector<uint64_t>{inputSize1});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WG_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WI_L_1", ParameterRange(inputSize1));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_2", std::vector<uint64_t>{inputSize2});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WG_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WI_L_2", ParameterRange(inputSize2));

        tuner.AddParameter(kernel, "INPUT_SIZE_R_1", std::vector<uint64_t>{inputSize3});
        tuner.AddParameter(kernel, "L_CB_SIZE_R_1", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "P_CB_SIZE_R_1", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "NUM_WG_R_1", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "NUM_WI_R_1", ParameterRange(inputSize3));

        tuner.AddParameter(kernel, "L_REDUCTION", std::vector<uint64_t>{1});
        tuner.AddParameter(kernel, "P_WRITE_BACK", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "L_WRITE_BACK", std::vector<uint64_t>{2});

        tuner.AddConstraint(kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_2"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_R_1", "OCL_DIM_L_2"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_R_1"}, UnequalConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_2", "INPUT_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_2", "INPUT_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "INPUT_SIZE_L_2", "NUM_WG_L_2"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_R_1", "INPUT_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "INPUT_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "INPUT_SIZE_R_1", "NUM_WG_R_1"}, LessThanOrEqualCeilDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "L_CB_SIZE_R_1"}, NoPostInSecondKernelConstraint);

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7] * values[8];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7] * values[8];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 2) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 2) * values[7] * values[8];
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7];
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7];
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 2) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 2) * values[7];
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1]
                + static_cast<uint64_t>(values[2] == 0) * values[3]
                + static_cast<uint64_t>(values[4] == 0) * values[5];
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1]
                + static_cast<uint64_t>(values[2] == 1) * values[3]
                + static_cast<uint64_t>(values[4] == 1) * values[5];
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1]
                + static_cast<uint64_t>(values[2] == 2) * values[3]
                + static_cast<uint64_t>(values[4] == 2) * values[5];
        });
    }
    else if constexpr (activeSample == AtfSampleType::CCSD)
    {
#if KTT_CUDA_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("tc_1", kernelPath + "TcAbcdefGebcDfga1.cu", ktt::DimensionVector(), ktt::DimensionVector());
        definition2 = tuner.AddKernelDefinitionFromFile("tc_2", kernelPath + "TcAbcdefGebcDfga2.cu", ktt::DimensionVector(), ktt::DimensionVector());
#elif KTT_OPENCL_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("tc_1", kernelPath + "TcAbcdefGebcDfga1.cl", ktt::DimensionVector(), ktt::DimensionVector());
        definition2 = tuner.AddKernelDefinitionFromFile("tc_2", kernelPath + "TcAbcdefGebcDfga2.cl", ktt::DimensionVector(), ktt::DimensionVector());
#endif

        std::vector<float> a(inputSize7 * inputSize5 * inputSize2 * inputSize3);
        std::vector<float> b(inputSize4 * inputSize6 * inputSize7 * inputSize1);
        std::vector<float> c(inputSize1 * inputSize2 * inputSize3 * inputSize4 * inputSize5 * inputSize6);
        std::vector<float> intRes(inputSize1 * inputSize2 * inputSize3 * inputSize4 * inputSize5 * inputSize6);
        std::vector<float> res(inputSize1 * inputSize2 * inputSize3 * inputSize4 * inputSize5 * inputSize6);
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            a[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < b.size(); ++i)
        {
            b[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < c.size(); ++i)
        {
            c[i] = 0.0f;
            intRes[i] = 0.0f;
            res[i] = 0.0f;
        }
        
        const size_t resSize = res.size() * sizeof(float);
        const auto aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
        const auto bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
        const auto cId = tuner.AddArgumentVector(c, ktt::ArgumentAccessType::ReadWrite);
        const auto resId = tuner.AddArgumentVector(res, ktt::ArgumentAccessType::ReadWrite);
        const auto intResId = tuner.AddArgumentVector(intRes, ktt::ArgumentAccessType::ReadWrite);
        tuner.SetArguments(definition, {aId, bId, resId, intResId});
        tuner.SetArguments(definition2, {intResId, resId, cId});

        kernel = tuner.CreateCompositeKernel("CCSD", {definition, definition2}, [resSize, resId, intResId, definition, definition2]
            (ktt::ComputeInterface& interface)
        {
            const auto& pairs = interface.GetCurrentConfiguration().GetPairs();
            size_t newResSize = resSize;

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "G_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            }

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1");
            }

            interface.ResizeBuffer(resId, newResSize, false);

            const size_t newIntResSize = resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            interface.ResizeBuffer(intResId, newIntResSize, false);

            interface.RunKernel(definition);

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1") > 1)
            {
                if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
                {
                    interface.ResizeBuffer(resId, resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1"), false);
                }

                interface.RunKernel(definition2);
            }
        });

        tuner.AddParameter(kernel, "CACHE_L_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "CACHE_P_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "G_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "L_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});
        tuner.AddParameter(kernel, "P_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});

        tuner.AddParameter(kernel, "OCL_DIM_L_1", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_L_2", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_L_3", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_L_4", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_L_5", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_L_6", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});
        tuner.AddParameter(kernel, "OCL_DIM_R_1", std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6});

        tuner.AddParameter(kernel, "INPUT_SIZE_L_1", std::vector<uint64_t>{inputSize1});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WG_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WI_L_1", ParameterRange(inputSize1));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_2", std::vector<uint64_t>{inputSize2});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WG_L_2", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WI_L_2", ParameterRange(inputSize2));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_3", std::vector<uint64_t>{inputSize3});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_3", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_3", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "NUM_WG_L_3", ParameterRange(inputSize3));
        tuner.AddParameter(kernel, "NUM_WI_L_3", ParameterRange(inputSize3));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_4", std::vector<uint64_t>{inputSize4});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_4", ParameterRange(inputSize4));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_4", ParameterRange(inputSize4));
        tuner.AddParameter(kernel, "NUM_WG_L_4", ParameterRange(inputSize4));
        tuner.AddParameter(kernel, "NUM_WI_L_4", ParameterRange(inputSize4));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_5", std::vector<uint64_t>{inputSize5});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_5", ParameterRange(inputSize5));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_5", ParameterRange(inputSize5));
        tuner.AddParameter(kernel, "NUM_WG_L_5", ParameterRange(inputSize5));
        tuner.AddParameter(kernel, "NUM_WI_L_5", ParameterRange(inputSize5));

        tuner.AddParameter(kernel, "INPUT_SIZE_L_6", std::vector<uint64_t>{inputSize6});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_6", ParameterRange(inputSize6));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_6", ParameterRange(inputSize6));
        tuner.AddParameter(kernel, "NUM_WG_L_6", ParameterRange(inputSize6));
        tuner.AddParameter(kernel, "NUM_WI_L_6", ParameterRange(inputSize6));

        tuner.AddParameter(kernel, "INPUT_SIZE_R_1", std::vector<uint64_t>{inputSize7});
        tuner.AddParameter(kernel, "L_CB_SIZE_R_1", ParameterRange(inputSize7));
        tuner.AddParameter(kernel, "P_CB_SIZE_R_1", ParameterRange(inputSize7));
        tuner.AddParameter(kernel, "NUM_WG_R_1", ParameterRange(inputSize7));
        tuner.AddParameter(kernel, "NUM_WI_R_1", ParameterRange(inputSize7));

        tuner.AddParameter(kernel, "L_REDUCTION", std::vector<uint64_t>{1});
        tuner.AddParameter(kernel, "P_WRITE_BACK", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "L_WRITE_BACK", std::vector<uint64_t>{6});

        tuner.AddConstraint(kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_2"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_3"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_4"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_5"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_L_6"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_R_1"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_2", "OCL_DIM_L_3"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_2", "OCL_DIM_L_4"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_2", "OCL_DIM_L_5"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_2", "OCL_DIM_L_6"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_2", "OCL_DIM_R_1"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_3", "OCL_DIM_L_4"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_3", "OCL_DIM_L_5"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_3", "OCL_DIM_L_6"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_3", "OCL_DIM_R_1"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_4", "OCL_DIM_L_5"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_4", "OCL_DIM_L_6"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_4", "OCL_DIM_R_1"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_5", "OCL_DIM_L_6"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_5", "OCL_DIM_R_1"}, UnequalConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_6", "OCL_DIM_R_1"}, UnequalConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_2", "INPUT_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_2", "INPUT_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_2", "INPUT_SIZE_L_2", "NUM_WG_L_2"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_3", "INPUT_SIZE_L_3"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_3", "L_CB_SIZE_L_3"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_3", "INPUT_SIZE_L_3", "L_CB_SIZE_L_3"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_3", "L_CB_SIZE_L_3", "P_CB_SIZE_L_3"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_3", "INPUT_SIZE_L_3", "NUM_WG_L_3"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_4", "INPUT_SIZE_L_4"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_4", "L_CB_SIZE_L_4"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_4", "INPUT_SIZE_L_4", "L_CB_SIZE_L_4"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_4", "L_CB_SIZE_L_4", "P_CB_SIZE_L_4"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_4", "INPUT_SIZE_L_4", "NUM_WG_L_4"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_5", "INPUT_SIZE_L_5"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_5", "L_CB_SIZE_L_5"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_5", "INPUT_SIZE_L_5", "L_CB_SIZE_L_5"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_5", "L_CB_SIZE_L_5", "P_CB_SIZE_L_5"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_5", "INPUT_SIZE_L_5", "NUM_WG_L_5"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_6", "INPUT_SIZE_L_6"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_6", "L_CB_SIZE_L_6"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_6", "INPUT_SIZE_L_6", "L_CB_SIZE_L_6"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_6", "L_CB_SIZE_L_6", "P_CB_SIZE_L_6"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_6", "INPUT_SIZE_L_6", "NUM_WG_L_6"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_R_1", "INPUT_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "INPUT_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "INPUT_SIZE_R_1", "NUM_WG_R_1"}, LessThanOrEqualCeilDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "L_CB_SIZE_R_1"}, NoPostInSecondKernelConstraint);

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7] * values[8]
                + static_cast<uint64_t>(values[9] == 0) * values[10] * values[11]
                + static_cast<uint64_t>(values[12] == 0) * values[13] * values[14]
                + static_cast<uint64_t>(values[15] == 0) * values[16] * values[17]
                + static_cast<uint64_t>(values[18] == 0) * values[19] * values[20];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7] * values[8]
                + static_cast<uint64_t>(values[9] == 1) * values[10] * values[11]
                + static_cast<uint64_t>(values[12] == 1) * values[13] * values[14]
                + static_cast<uint64_t>(values[15] == 1) * values[16] * values[17]
                + static_cast<uint64_t>(values[18] == 1) * values[19] * values[20];
        });

        tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return std::max(static_cast<uint64_t>(values[0] >= 2) * values[1] * values[2], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[3] >= 2) * values[4] * values[5], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[6] >= 2) * values[7] * values[8], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[9] >= 2) * values[10] * values[11], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[12] >= 2) * values[13] * values[14], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[15] >= 2) * values[16] * values[17], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[18] >= 2) * values[19] * values[20], static_cast<uint64_t>(1));
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7] * values[8]
                + static_cast<uint64_t>(values[9] == 0) * values[10] * values[11]
                + static_cast<uint64_t>(values[12] == 0) * values[13] * values[14]
                + static_cast<uint64_t>(values[15] == 0) * values[16] * values[17]
                + static_cast<uint64_t>(values[18] == 0) * values[19];
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7] * values[8]
                + static_cast<uint64_t>(values[9] == 1) * values[10] * values[11]
                + static_cast<uint64_t>(values[12] == 1) * values[13] * values[14]
                + static_cast<uint64_t>(values[15] == 1) * values[16] * values[17]
                + static_cast<uint64_t>(values[18] == 1) * values[19];
        });

        tuner.AddThreadModifier(kernel, {definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WG_L_3", "NUM_WI_L_3",
            "OCL_DIM_L_4", "NUM_WG_L_4", "NUM_WI_L_4", "OCL_DIM_L_5", "NUM_WG_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WG_L_6", "NUM_WI_L_6",
            "OCL_DIM_R_1", "NUM_WI_R_1"}, [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return std::max(static_cast<uint64_t>(values[0] >= 2) * values[1] * values[2], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[3] >= 2) * values[4] * values[5], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[6] >= 2) * values[7] * values[8], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[9] >= 2) * values[10] * values[11], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[12] >= 2) * values[13] * values[14], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[15] >= 2) * values[16] * values[17], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[18] >= 2) * values[19], static_cast<uint64_t>(1));
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WI_L_3", "OCL_DIM_L_4", "NUM_WI_L_4",
            "OCL_DIM_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WI_L_6", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1]
                + static_cast<uint64_t>(values[2] == 0) * values[3]
                + static_cast<uint64_t>(values[4] == 0) * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7]
                + static_cast<uint64_t>(values[8] == 0) * values[9]
                + static_cast<uint64_t>(values[10] == 0) * values[11]
                + static_cast<uint64_t>(values[12] == 0) * values[13];
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WI_L_3", "OCL_DIM_L_4", "NUM_WI_L_4",
            "OCL_DIM_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WI_L_6", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1]
                + static_cast<uint64_t>(values[2] == 1) * values[3]
                + static_cast<uint64_t>(values[4] == 1) * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7]
                + static_cast<uint64_t>(values[8] == 1) * values[9]
                + static_cast<uint64_t>(values[10] == 1) * values[11]
                + static_cast<uint64_t>(values[12] == 1) * values[13];
        });

        tuner.AddThreadModifier(kernel, {definition, definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_L_3", "NUM_WI_L_3", "OCL_DIM_L_4", "NUM_WI_L_4",
            "OCL_DIM_L_5", "NUM_WI_L_5", "OCL_DIM_L_6", "NUM_WI_L_6", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const std::vector<uint64_t>& values)
        {
            return std::max(static_cast<uint64_t>(values[0] >= 2) * values[1], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[2] >= 2) * values[3], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[4] >= 2) * values[5], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[6] >= 2) * values[7], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[8] >= 2) * values[9], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[10] >= 2) * values[11], static_cast<uint64_t>(1))
                * std::max(static_cast<uint64_t>(values[12] >= 2) * values[13], static_cast<uint64_t>(1));
        });
    }
    else // AtfSampleType::PRL
    {
#if KTT_CUDA_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("rl_1", kernelPath + "Rl1.cu", ktt::DimensionVector(), ktt::DimensionVector());
#elif KTT_OPENCL_EXAMPLE
        definition = tuner.AddKernelDefinitionFromFile("rl_1", kernelPath + "Rl1.cl", ktt::DimensionVector(), ktt::DimensionVector());
#endif
        kernel = tuner.CreateSimpleKernel("PRL", definition);

        tuner.AddParameter(kernel, "CACHE_L_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "CACHE_P_CB", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "G_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2});
        tuner.AddParameter(kernel, "L_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});
        tuner.AddParameter(kernel, "P_CB_RES_DEST_LEVEL", std::vector<uint64_t>{2, 1, 0});

        tuner.AddParameter(kernel, "OCL_DIM_L_1", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernel, "OCL_DIM_R_1", std::vector<uint64_t>{0, 1});

        tuner.AddParameter(kernel, "INPUT_SIZE_L_1", std::vector<uint64_t>{inputSize1});
        tuner.AddParameter(kernel, "L_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "P_CB_SIZE_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WG_L_1", ParameterRange(inputSize1));
        tuner.AddParameter(kernel, "NUM_WI_L_1", ParameterRange(inputSize1));

        tuner.AddParameter(kernel, "INPUT_SIZE_R_1", std::vector<uint64_t>{inputSize2});
        tuner.AddParameter(kernel, "L_CB_SIZE_R_1", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "P_CB_SIZE_R_1", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WG_R_1", ParameterRange(inputSize2));
        tuner.AddParameter(kernel, "NUM_WI_R_1", ParameterRange(inputSize2));

        tuner.AddParameter(kernel, "L_REDUCTION", std::vector<uint64_t>{1});
        tuner.AddParameter(kernel, "P_WRITE_BACK", std::vector<uint64_t>{0});
        tuner.AddParameter(kernel, "L_WRITE_BACK", std::vector<uint64_t>{1});

        tuner.AddConstraint(kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        tuner.AddConstraint(kernel, {"OCL_DIM_L_1", "OCL_DIM_R_1"}, UnequalConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        tuner.AddConstraint(kernel, {"L_CB_SIZE_R_1", "INPUT_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"P_CB_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "INPUT_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1"}, DividesDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WI_R_1", "INPUT_SIZE_R_1", "NUM_WG_R_1"}, LessThanOrEqualCeilDivConstraint);
        tuner.AddConstraint(kernel, {"NUM_WG_R_1", "L_CB_SIZE_R_1"}, NoPostInSecondKernelConstraint);

        // Only search space generation is supported in this sample
        tuner.Tune(kernel);
        return 0;
    }

    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    //auto results = tuner.Tune(kernel, std::make_unique<ktt::ConfigurationCount>(100));
    tuner.SaveResults(results, "AtfOutput", ktt::OutputFormat::XML);
    return 0;
}
