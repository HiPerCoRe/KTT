#include "../ExampleReferenceKernel.h"
#include "Api/Configuration/DimensionVector.h"
#include <memory>

using namespace std;

bool IsMultiple(const size_t a, const size_t b)
{
    return a % b == 0;
};

class ClTuneGemm : public ExampleReferenceKernel {
protected:
    ClTuneGemm(int argc, char **argv, string exampleFolderPath,
               string defaultKernelFileBaseName, string defaultRefKernelFileBaseName) :
        ExampleReferenceKernel(argc, argv, exampleFolderPath,
                               defaultKernelFileBaseName, defaultRefKernelFileBaseName),
        m_kSizeM(4 * 1024),
        m_kSizeN(4 * 1024),
        m_kSizeK(4 * 1024),
        m_gridDimensions(m_kSizeM, m_kSizeN)
    {
    }

    friend ExampleReferenceKernel;

    uint32_t m_kSizeM;
    uint32_t m_kSizeN;
    uint32_t m_kSizeK;

    ktt::DimensionVector m_gridDimensions;

    vector<float> m_matA;
    vector<float> m_matB;
    vector<float> m_matC;

    ktt::ArgumentId m_kSizeMId;
    ktt::ArgumentId m_kSizeNId;
    ktt::ArgumentId m_kSizeKId;
    ktt::ArgumentId m_matAId;
    ktt::ArgumentId m_matBId;
    ktt::ArgumentId m_matCId;

    void InitCLI() override
    {
        ExampleBase::InitCLI();

        m_cli.AddOption({[this](const vector<string> &args){
                m_kSizeM = stoi(args[0]);
                m_kSizeN = stoi(args[1]);
                m_kSizeK = stoi(args[2]);
            }, "--matSizes", "Sets matrix sizes (expects 3 ints) for computation C = A x B. Matrix sizes will be:"
            "\n\tA: M x K\n\tB: K x N\n\tC: M x N", "<M> <N> <K>", 3
        });
    }

    void InitData() override
    {
        m_matA.resize(m_kSizeM * m_kSizeK);
        m_matB.resize(m_kSizeN * m_kSizeK);
        m_matC.resize(m_kSizeM * m_kSizeN, 0.0f);

        FillBuffers<float>({&m_matA, &m_matB}, -2.0f, 2.0f);
    }

    void InitKernel() override
    {
        m_gridDimensions = ktt::DimensionVector(m_kSizeM, m_kSizeN);
        m_kSizeMId = m_tuner->AddArgumentScalar(m_kSizeM);
        m_kSizeNId = m_tuner->AddArgumentScalar(m_kSizeN);
        m_kSizeKId = m_tuner->AddArgumentScalar(m_kSizeK);
        m_matAId = m_tuner->AddArgumentVector(m_matA, ktt::ArgumentAccessType::ReadOnly);
        m_matBId = m_tuner->AddArgumentVector(m_matB, ktt::ArgumentAccessType::ReadOnly);
        m_matCId = m_tuner->AddArgumentVector(m_matC, ktt::ArgumentAccessType::WriteOnly);

        InitKernelDefault("gemm_fast", "Gemm", m_gridDimensions,
            {m_kSizeMId, m_kSizeNId, m_kSizeKId, m_matAId, m_matBId, m_matCId});
    }

    void InitTuningSpace() override
    {
        m_tuner->AddParameter(m_kernel, "MWG", vector<uint64_t>{16, 32, 64, 128});
        m_tuner->AddParameter(m_kernel, "NWG", vector<uint64_t>{16, 32, 64, 128});
        m_tuner->AddParameter(m_kernel, "KWG", vector<uint64_t>{16, 32});
        m_tuner->AddParameter(m_kernel, "MDIMC", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "NDIMC", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "MDIMA", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "NDIMB", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "KWI", vector<uint64_t>{2, 8});

        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner->AddParameter(m_kernel, "VWM", vector<uint64_t>{1, 2, 4, 8});
            m_tuner->AddParameter(m_kernel, "VWN", vector<uint64_t>{1, 2, 4, 8});
        }
        else
        {
            m_tuner->AddParameter(m_kernel, "VWM", vector<uint64_t>{1, 2, 4});
            m_tuner->AddParameter(m_kernel, "VWN", vector<uint64_t>{1, 2, 4});
        }

        m_tuner->AddParameter(m_kernel, "STRM", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "STRN", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "SA", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "SB", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "PRECISION", vector<uint64_t>{32});

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "MWG", ktt::ModifierAction::Divide);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "NWG", ktt::ModifierAction::Divide);

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "MDIMC", ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "NDIMC", ktt::ModifierAction::Multiply);

        // Add conditions
        // Sets constraints: Set-up the constraints functions to use. The constraints require a function
        // object (in this case a lambda) which takes a vector of tuning parameter values and returns
        // a boolean value whether or not the tuning configuration is legal. In this case, the helper
        // function 'IsMultiple' is employed for convenience. In the calls to 'AddConstraint' below, the
        // vector of parameter names (as strings) matches the input integer vector of the lambda's.
        auto multipleOfX = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1]);};
        auto multipleOfXMulY = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], v[1] * v[2]);};
        auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v) {return IsMultiple(v[0], (v[1] * v[2]) / v[3]);};

        m_tuner->AddConstraint(m_kernel, {"KWG", "KWI"}, multipleOfX);
        m_tuner->AddConstraint(m_kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
        m_tuner->AddConstraint(m_kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);
    }

    void InitReference() override
    {
        ktt::DimensionVector m_referenceGridDimensions = m_gridDimensions;
        const ktt::DimensionVector m_referenceBlockDimensions{8, 8};
        m_referenceGridDimensions.Divide(m_referenceBlockDimensions);
        InitReferenceKernelDefault("gemm_reference", m_referenceGridDimensions, m_referenceBlockDimensions,
            {m_kSizeMId, m_kSizeNId, m_kSizeKId, m_matAId, m_matBId, m_matCId},
            {m_matCId}, 0.001);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<ClTuneGemm> clTuneGemm = ClTuneGemm::Create<ClTuneGemm>(argc, argv, "Examples/ClTuneGemm",
        "ClTuneGemm", "ClTuneGemmReference");
    clTuneGemm->Run();

    return 0;
}
