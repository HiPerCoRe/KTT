#include "../ExampleBase.h"
#include <cstdint>
#include <memory>

using namespace std;

class AtfPRL : public ExampleBase {
protected:
    AtfPRL(int argc, char **argv, int defaultProblemSize,
           string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        // Keep OpenCL sizes as specified
        m_inputSize1 = static_cast<uint64_t>(sqrt(m_problemSize)) * 1024;
        m_inputSize2 = m_inputSize1;

        m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    }

    friend ExampleBase;

    // Input sizes - kept as member variables
    uint64_t m_inputSize1;
    uint64_t m_inputSize2;

    // Data vectors
    vector<float> m_a;
    vector<float> m_b;

    // Argument IDs
    ktt::ArgumentId m_aId;
    ktt::ArgumentId m_bId;

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
        m_a.resize(m_inputSize1 * m_inputSize1);
        m_b.resize(m_inputSize2 * m_inputSize2);

        for (size_t i = 0; i < m_a.size(); ++i)
        {
            m_a[i] = static_cast<float>((i % 100) + 1);
        }

        for (size_t i = 0; i < m_b.size(); ++i)
        {
            m_b[i] = static_cast<float>((i % 100) + 1);
        }
    }

    void InitKernel() override
    {
        m_aId = m_tuner->AddArgumentVector(m_a, ktt::ArgumentAccessType::ReadOnly);
        m_bId = m_tuner->AddArgumentVector(m_b, ktt::ArgumentAccessType::ReadWrite);

        InitKernelDefault("rl_1", "PRL", ktt::DimensionVector(), {m_aId, m_bId});
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

        m_tuner->AddParameter(m_kernel, "OCL_DIM_L_1", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "OCL_DIM_R_1", vector<uint64_t>{0, 1});

        m_tuner->AddParameter(m_kernel, "INPUT_SIZE_L_1", vector<uint64_t>{m_inputSize1});
        m_tuner->AddParameter(m_kernel, "L_CB_SIZE_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "P_CB_SIZE_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "NUM_WG_L_1", ParameterRange(m_inputSize1));
        m_tuner->AddParameter(m_kernel, "NUM_WI_L_1", ParameterRange(m_inputSize1));

        m_tuner->AddParameter(m_kernel, "INPUT_SIZE_R_1", vector<uint64_t>{m_inputSize2});
        m_tuner->AddParameter(m_kernel, "L_CB_SIZE_R_1", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "P_CB_SIZE_R_1", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "NUM_WG_R_1", ParameterRange(m_inputSize2));
        m_tuner->AddParameter(m_kernel, "NUM_WI_R_1", ParameterRange(m_inputSize2));

        m_tuner->AddParameter(m_kernel, "L_REDUCTION", vector<uint64_t>{1});
        m_tuner->AddParameter(m_kernel, "P_WRITE_BACK", vector<uint64_t>{0});
        m_tuner->AddParameter(m_kernel, "L_WRITE_BACK", vector<uint64_t>{1});

        // Add constraints
        m_tuner->AddConstraint(m_kernel, {"G_CB_RES_DEST_LEVEL", "L_CB_RES_DEST_LEVEL", "P_CB_RES_DEST_LEVEL"}, DescendingConstraint);
        m_tuner->AddConstraint(m_kernel, {"OCL_DIM_L_1", "OCL_DIM_R_1"}, UnequalConstraint);

        m_tuner->AddConstraint(m_kernel, {"L_CB_SIZE_L_1", "INPUT_SIZE_L_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"P_CB_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_L_1", "INPUT_SIZE_L_1", "L_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_L_1", "INPUT_SIZE_L_1", "NUM_WG_L_1"}, LessThanOrEqualCeilDivConstraint);

        m_tuner->AddConstraint(m_kernel, {"L_CB_SIZE_R_1", "INPUT_SIZE_R_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"P_CB_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_R_1", "INPUT_SIZE_R_1", "L_CB_SIZE_R_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1"}, DividesDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WI_R_1", "INPUT_SIZE_R_1", "NUM_WG_R_1"}, LessThanOrEqualCeilDivConstraint);
        m_tuner->AddConstraint(m_kernel, {"NUM_WG_R_1", "L_CB_SIZE_R_1"}, NoPostInSecondKernelConstraint);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<AtfPRL> atfPRL = AtfPRL::Create<AtfPRL>(argc, argv, 1, "Examples/AtfPRL", "Rl1");
    atfPRL->Run();

    return 0;
}
