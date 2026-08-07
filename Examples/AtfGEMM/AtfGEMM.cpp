#include "../ExampleBase.h"
#include <memory>

using namespace std;

class AtfGEMM : public ExampleBase {
protected:
    AtfGEMM(int argc, char **argv, int defaultProblemSize,
            string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        // Keep OpenCL sizes as specified
        m_inputSize1 = static_cast<uint64_t>(sqrt(m_problemSize)) * 1024;
        m_inputSize2 = m_inputSize1;
        m_inputSize3 = m_inputSize1;

        m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);

        m_kernelPath1 = GetKernelFilePath(exampleFolderPath, "Gemm1");
        m_kernelPath2 = GetKernelFilePath(exampleFolderPath, "Gemm2");
    }

    friend ExampleBase;

    // Input sizes - kept as member variables
    uint64_t m_inputSize1;
    uint64_t m_inputSize2;
    uint64_t m_inputSize3;

    // Kernel paths
    std::string m_kernelPath1;
    std::string m_kernelPath2;

    // Data vectors
    vector<float> m_a;
    vector<float> m_b;
    vector<float> m_c;
    vector<float> m_intRes;
    vector<float> m_res;

    // Argument IDs
    ktt::ArgumentId m_aId;
    ktt::ArgumentId m_bId;
    ktt::ArgumentId m_cId;
    ktt::ArgumentId m_intResId;
    ktt::ArgumentId m_resId;

    // Kernel definitions
    ktt::KernelDefinitionId m_definition;
    ktt::KernelDefinitionId m_definition2;

    size_t m_resSize;

    // Helper function for parameter range
    vector<uint64_t> ParameterRange(const uint64_t max)
    {
        vector<uint64_t> values;

        for (uint64_t i = 1; i <= max; ++i)
        {
            values.push_back(i);
        }

        return values;
    }

    void InitData() override
    {
        // Initialize data buffers with fixed sizes
        m_a.resize(m_inputSize1 * m_inputSize3);
        m_b.resize(m_inputSize3 * m_inputSize2);
        m_c.resize(m_inputSize1 * m_inputSize2);
        m_intRes.resize(m_inputSize1 * m_inputSize2);
        m_res.resize(m_inputSize1 * m_inputSize2);

        for (size_t i = 0; i < m_a.size(); ++i)
        {
            m_a[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < m_b.size(); ++i)
        {
            m_b[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < m_c.size(); ++i)
        {
            m_c[i] = 0.0f;
            m_intRes[i] = 0.0f;
            m_res[i] = 0.0f;
        }

        m_resSize = m_res.size() * sizeof(float);
    }

    void InitKernel() override
    {
        m_aId = m_tuner->AddArgumentVector(m_a, ktt::ArgumentAccessType::ReadOnly);
        m_bId = m_tuner->AddArgumentVector(m_b, ktt::ArgumentAccessType::ReadOnly);
        m_cId = m_tuner->AddArgumentVector(m_c, ktt::ArgumentAccessType::ReadWrite);
        m_intResId = m_tuner->AddArgumentVector(m_intRes, ktt::ArgumentAccessType::ReadWrite);
        m_resId = m_tuner->AddArgumentVector(m_res, ktt::ArgumentAccessType::ReadWrite);

        m_definition = m_tuner->AddKernelDefinitionFromFile("gemm_1", m_kernelPath1, ktt::DimensionVector(), ktt::DimensionVector());
        m_definition2 = m_tuner->AddKernelDefinitionFromFile("gemm_2", m_kernelPath2, ktt::DimensionVector(), ktt::DimensionVector());

        m_tuner->SetArguments(m_definition, {m_aId, m_bId, m_resId, m_intResId});
        m_tuner->SetArguments(m_definition2, {m_intResId, m_resId, m_cId});

        m_kernel = m_tuner->CreateCompositeKernel("GEMM", {m_definition, m_definition2}, [this](ktt::ComputeInterface& interface)
        {
            const auto& pairs = interface.GetCurrentConfiguration().GetPairs();
            size_t newResSize = m_resSize;

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "G_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            }

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
            {
                newResSize *= ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1");
            }

            interface.ResizeBuffer(m_resId, newResSize, false);

            const size_t newIntResSize = m_resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1");
            interface.ResizeBuffer(m_intResId, newIntResSize, false);

            if (m_computeApi == ktt::ComputeApi::CUDA && false)
                interface.RunKernelWithProfiling(m_definition);
            else
                interface.RunKernel(m_definition);

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WG_R_1") > 1)
            {
                if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "L_CB_RES_DEST_LEVEL") == 2)
                {
                    interface.ResizeBuffer(m_resId, m_resSize * ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "NUM_WI_R_1"), false);
                }

                interface.RunKernel(m_definition2);
            }
        });
    }

    void InitTuningSpace() override
    {
        // Constraint lambdas
        auto DescendingConstraint = [](const vector<uint64_t>& v)
        {
            bool valid = true;

            for (size_t i = 1; i < v.size(); ++i)
            {
                valid = valid && (v[i - 1] >= v[i]);
            }

            return valid;
        };

        auto UnequalConstraint = [](const vector<uint64_t>& v)
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

        auto LessThanOrEqualCeilDivConstraint = [](const vector<uint64_t>& v) { return v[0] <= (v[1] + v[2] - 1) / v[2]; };
        auto DividesConstraint = [](const vector<uint64_t>& v) { return v[1] % v[0] == 0; };
        auto DividesDivConstraint = [](const vector<uint64_t>& v) { return (v[1] / v[2]) % v[0] == 0; };
        auto NoPostInSecondKernelConstraint = [](const vector<uint64_t>& v) { return v[0] == 1 || (v[0] % v[1] == 0); };

        // Add parameters
        m_tuner->AddParameter(m_kernel, "CACHE_L_CB", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "CACHE_P_CB", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "G_CB_RES_DEST_LEVEL", vector<uint64_t>{2});
        m_tuner->AddParameter(m_kernel, "L_CB_RES_DEST_LEVEL", vector<uint64_t>{2, 1, 0});
        m_tuner->AddParameter(m_kernel, "P_CB_RES_DEST_LEVEL", vector<uint64_t>{2, 1, 0});

        m_tuner->AddParameter(m_kernel, "OCL_DIM_L_1", vector<uint64_t>{0, 1, 2});
        m_tuner->AddParameter(m_kernel, "OCL_DIM_L_2", vector<uint64_t>{0, 1, 2});
        m_tuner->AddParameter(m_kernel, "OCL_DIM_R_1", vector<uint64_t>{0, 1, 2});

        m_tuner->AddParameter(m_kernel, "INPUT_SIZE_L_1", vector<uint64_t>{m_inputSize1});
        m_tuner->AddParameter(m_kernel, "L_CB_SIZE_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "P_CB_SIZE_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "NUM_WG_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "NUM_WI_L_1", ParameterRange(m_inputSize1));

        m_tuner->AddParameter(m_kernel, "INPUT_SIZE_L_2", vector<uint64_t>{m_inputSize2});
        m_tuner->AddParameter(m_kernel, "L_CB_SIZE_L_2", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "P_CB_SIZE_L_2", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "NUM_WG_L_2", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "NUM_WI_L_2", ParameterRange(m_inputSize2));

        m_tuner->AddParameter(m_kernel, "INPUT_SIZE_R_1", vector<uint64_t>{m_inputSize3});
        m_tuner->AddParameter(m_kernel, "L_CB_SIZE_R_1", ParameterRange(m_inputSize3));
        m_tuner->AddParameter(m_kernel, "P_CB_SIZE_R_1", ParameterRange(m_inputSize3));
        m_tuner->AddParameter(m_kernel, "NUM_WG_R_1", ParameterRange(m_inputSize3));
        m_tuner->AddParameter(m_kernel, "NUM_WI_R_1", ParameterRange(m_inputSize3));

        m_tuner->AddParameter(m_kernel, "L_REDUCTION", vector<uint64_t>{1});
        m_tuner->AddParameter(m_kernel, "P_WRITE_BACK", vector<uint64_t>{0});
        m_tuner->AddParameter(m_kernel, "L_WRITE_BACK", vector<uint64_t>{2});

        // Add constraints
        m_tuner->AddConstraint(m_kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        m_tuner->AddConstraint(m_kernel, {"OCL_DIM_L_1", "OCL_DIM_L_2"}, UnequalConstraint);
        m_tuner->AddConstraint(m_kernel, {"OCL_DIM_R_1", "OCL_DIM_L_2"}, UnequalConstraint);
        m_tuner->AddConstraint(m_kernel, {"OCL_DIM_L_1", "OCL_DIM_R_1"}, UnequalConstraint);

        m_tuner->AddConstraint(m_kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        m_tuner->AddConstraint(m_kernel, {"L_CB_SIZE_L_2", "INPUT_SIZE_L_2"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"P_CB_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_L_2", "INPUT_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_2", "INPUT_SIZE_L_2", "NUM_WG_L_2"}, LessThanOrEqualCeilDivConstraint);

        m_tuner->AddConstraint(m_kernel, {"L_CB_SIZE_R_1", "INPUT_SIZE_R_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"P_CB_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_R_1", "INPUT_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_R_1", "INPUT_SIZE_R_1", "NUM_WG_R_1"}, LessThanOrEqualCeilDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_R_1", "L_CB_SIZE_R_1"}, NoPostInSecondKernelConstraint);

        // Thread modifiers for first kernel (definition)
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7] * values[8];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7] * values[8];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WG_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 2) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 2) * values[7] * values[8];
        });

        // Thread modifiers for second kernel (definition2)
        m_tuner->AddThreadModifier(m_kernel, {m_definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 0) * values[7];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 1) * values[7];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition2}, ktt::ModifierType::Global, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 2) * values[4] * values[5]
                + static_cast<uint64_t>(values[6] == 2) * values[7];
        });

        // Local thread modifiers
        m_tuner->AddThreadModifier(m_kernel, {m_definition, m_definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1]
                + static_cast<uint64_t>(values[2] == 0) * values[3]
                + static_cast<uint64_t>(values[4] == 0) * values[5];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition, m_definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1]
                + static_cast<uint64_t>(values[2] == 1) * values[3]
                + static_cast<uint64_t>(values[4] == 1) * values[5];
        });

        m_tuner->AddThreadModifier(m_kernel, {m_definition, m_definition2}, ktt::ModifierType::Local, ktt::ModifierDimension::Z,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2", "OCL_DIM_R_1", "NUM_WI_R_1"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 2) * values[1]
                + static_cast<uint64_t>(values[2] == 2) * values[3]
                + static_cast<uint64_t>(values[4] == 2) * values[5];
        });
    }
};

int main(int argc, char **argv)
{
    unique_ptr<AtfGEMM> atfGEMM = AtfGEMM::Create<AtfGEMM>(argc, argv, 4, "Examples/AtfGEMM", "AtfGEMM");
    atfGEMM->Run();

    return 0;
}
