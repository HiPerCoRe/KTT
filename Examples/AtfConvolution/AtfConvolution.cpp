#include "../ExampleBase.h"
#include <memory>

using namespace std;

class AtfConvolution : public ExampleBase {
protected:
    AtfConvolution(shared_ptr<ExampleConfiguration> config, int defaultProblemSize,
                   string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        // Keep OpenCL sizes as specified
        m_inputSize1 = 4096;
        m_inputSize2 = 4096;

        m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    }

    friend ExampleBase;

    // Input sizes - kept as member variables
    uint64_t m_inputSize1;
    uint64_t m_inputSize2;

    // Data vectors
    vector<float> m_in;
    vector<float> m_out;
    vector<float> m_intRes;

    // Argument IDs
    ktt::ArgumentId m_inId;
    ktt::ArgumentId m_outId;
    ktt::ArgumentId m_intResId;

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
        m_in.resize(m_inputSize1 * m_inputSize2);
        m_out.resize((m_inputSize1 - 4) * (m_inputSize2 - 4));
        m_intRes.resize((m_inputSize1 - 4) * (m_inputSize2 - 4));

        for (size_t i = 0; i < m_in.size(); ++i)
        {
            m_in[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < m_out.size(); ++i)
        {
            m_out[i] = 0.0f;
        }

        for (size_t i = 0; i < m_intRes.size(); ++i)
        {
            m_intRes[i] = 0.0f;
        }
    }

    void InitKernel() override
    {
        m_inId = m_tuner.AddArgumentVector(m_in, ktt::ArgumentAccessType::ReadOnly);
        m_outId = m_tuner.AddArgumentVector(m_out, ktt::ArgumentAccessType::ReadWrite);
        m_intResId = m_tuner.AddArgumentVector(m_intRes, ktt::ArgumentAccessType::ReadWrite);

        InitKernelDefault("gaussian_1", "Convolution", ktt::DimensionVector(), {m_inId, m_outId, m_intResId});
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

        // Add parameters
        m_tuner.AddParameter(m_kernel, "CACHE_L_CB", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "CACHE_P_CB", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "G_CB_RES_DEST_LEVEL", vector<uint64_t>{2});
        m_tuner.AddParameter(m_kernel, "L_CB_RES_DEST_LEVEL", vector<uint64_t>{2, 1, 0});
        m_tuner.AddParameter(m_kernel, "P_CB_RES_DEST_LEVEL", vector<uint64_t>{2, 1, 0});

        m_tuner.AddParameter(m_kernel, "OCL_DIM_L_1", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "OCL_DIM_L_2", vector<uint64_t>{0, 1});

        m_tuner.AddParameter(m_kernel, "INPUT_SIZE_L_1", vector<uint64_t>{m_inputSize1 - 4});
        m_tuner.AddParameter(m_kernel, "L_CB_SIZE_L_1", ParameterRange(m_inputSize1 - 4));
        m_tuner.AddParameter(m_kernel, "P_CB_SIZE_L_1", ParameterRange(m_inputSize1 - 4));
        m_tuner.AddParameter(m_kernel, "NUM_WG_L_1", ParameterRange(m_inputSize1 - 4));
        m_tuner.AddParameter(m_kernel, "NUM_WI_L_1", ParameterRange(m_inputSize1 - 4));

        m_tuner.AddParameter(m_kernel, "INPUT_SIZE_L_2", vector<uint64_t>{m_inputSize2 - 4});
        m_tuner.AddParameter(m_kernel, "L_CB_SIZE_L_2", ParameterRange(m_inputSize2 - 4));
        m_tuner.AddParameter(m_kernel, "P_CB_SIZE_L_2", ParameterRange(m_inputSize2 - 4));
        m_tuner.AddParameter(m_kernel, "NUM_WG_L_2", ParameterRange(m_inputSize2 - 4));
        m_tuner.AddParameter(m_kernel, "NUM_WI_L_2", ParameterRange(m_inputSize2 - 4));

        m_tuner.AddParameter(m_kernel, "L_REDUCTION", vector<uint64_t>{1});
        m_tuner.AddParameter(m_kernel, "P_WRITE_BACK", vector<uint64_t>{0});
        m_tuner.AddParameter(m_kernel, "L_WRITE_BACK", vector<uint64_t>{2});

        // Add constraints
        m_tuner.AddConstraint(m_kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        m_tuner.AddConstraint(m_kernel, {"OCL_DIM_L_1", "OCL_DIM_L_2"}, UnequalConstraint);

        m_tuner.AddConstraint(m_kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        m_tuner.AddConstraint(m_kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        m_tuner.AddConstraint(m_kernel, {"L_CB_SIZE_L_2", "INPUT_SIZE_L_2"}, DividesConstraint);
        m_tuner.AddConstraint(m_kernel, {"P_CB_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WG_L_2", "INPUT_SIZE_L_2", "L_CB_SIZE_L_2"}, DividesDivConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WI_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2"}, DividesDivConstraint);
        m_tuner.AddConstraint(m_kernel, {"NUM_WI_L_2", "INPUT_SIZE_L_2", "NUM_WG_L_2"}, LessThanOrEqualCeilDivConstraint);

        // Thread modifiers for global X dimension
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 0) * values[4] * values[5];
        });

        // Thread modifiers for global Y dimension
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WG_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WG_L_2", "NUM_WI_L_2"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1] * values[2]
                + static_cast<uint64_t>(values[3] == 1) * values[4] * values[5];
        });

        // Thread modifiers for local X dimension
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 0) * values[1]
                + static_cast<uint64_t>(values[2] == 0) * values[3];
        });

        // Thread modifiers for local Y dimension
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y,
            {"OCL_DIM_L_1", "NUM_WI_L_1", "OCL_DIM_L_2", "NUM_WI_L_2"},
            [](const uint64_t, const vector<uint64_t>& values)
        {
            return static_cast<uint64_t>(values[0] == 1) * values[1]
                + static_cast<uint64_t>(values[2] == 1) * values[3];
        });
    }
};

int main(int argc, char **argv)
{
    unique_ptr<AtfConvolution> atfConvolution = AtfConvolution::Create<AtfConvolution>(argc, argv, 1, "Examples/AtfConvolution", "GaussianStatic1");
    atfConvolution->Run();

    return 0;
}
