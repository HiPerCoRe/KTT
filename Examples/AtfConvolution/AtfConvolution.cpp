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
    //const std::string defaultMlModel = kernelPrefix + "../Examples/AtfSamples/Models/2080Ti-AtfGEMM_output_DT.sav"; //GEMM is composition, which cannot be fully profiled in KTT (only the first kernel is profiled & navigated, which does not work well)
    const std::string defaultMlModel = kernelPrefix + "../Examples/AtfSamples/Models/2080Ti-AtfConvolution_output_DT.sav";
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

    inputSize1 = 4096;
    inputSize2 = 4096;

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

    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    ktt::KernelDefinitionId definition;
    ktt::KernelId kernel;

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

    if constexpr (computeApi == ktt::ComputeApi::CUDA && useProfiling) {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }
#ifdef KTT_CUDA_EXAMPLE
    //tuner.SetProfileBasedSearcher(kernel, defaultMlModel);
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
#else
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
#endif
    auto results = tuner.Tune(kernel, std::make_unique<ktt::ConfigurationCount>(2));
    tuner.SaveResults(results, "AtfOutput", ktt::OutputFormat::JSON);
    tuner.SaveResults(results, "AtfOutput", ktt::OutputFormat::XML);
    return 0;
}
