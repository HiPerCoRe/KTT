#include "../ExampleReferenceComputation.h"
#include <memory>

using namespace std;

bool IsMultiple(size_t a, size_t b) { return a % b == 0; };

class Covariance : public ExampleReferenceComputation {
protected:
    Covariance(shared_ptr<ExampleConfiguration> config, int defaultProblemSize, string exampleFolderPath,
               string defaultKernelFileBaseName) :
        ExampleReferenceComputation(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName),
        // Covariance has O(n × m²) complexity. For square matrices where n ≈ m,
        // we scale with cube root of problem size to keep total work proportional
        m_n(static_cast<int>(sqrt(m_problemSize)) * 1024),
        m_m(static_cast<int>(sqrt(m_problemSize)) * 1024)
    {
        m_refKernelFile = GetKernelFilePath(exampleFolderPath, "CovarianceReference");
        m_gemmFile = GetKernelFilePath(exampleFolderPath, "Gemm");
        m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    }

    friend ExampleBase;

    int m_n;
    int m_m;

    const uint64_t MAX_WORK_GROUP_SIZE = 1024;

    vector<float> m_data;
    vector<float> m_symmat;
    vector<float> m_mean;

    ktt::ArgumentId mDataId;
    ktt::ArgumentId mSymmatId;
    ktt::ArgumentId mMeanId;
    ktt::ArgumentId mMId;
    ktt::ArgumentId mNId;
    ktt::ArgumentId mFloatNId;

    string m_refKernelFile;
    string m_gemmFile;

    ktt::KernelDefinitionId m_refMeanDefinition;
    ktt::KernelDefinitionId m_refReduceDefinition;
    ktt::KernelDefinitionId m_refCovarDefinition;

    ktt::KernelDefinitionId m_meanDefinition;
    ktt::KernelDefinitionId m_reduceDefinition;
    ktt::KernelDefinitionId m_covarDefinition;
    ktt::KernelDefinitionId m_gemmDefinition;
    ktt::KernelDefinitionId m_triangularToSymmetricDefinition;

    void InitData() override
    {
        m_data.resize(m_m * m_n);
        m_symmat.resize(m_m * m_m, 0.0f);
        m_mean.resize(m_m, 0.0f);

        FillBuffers<float>({&m_data}, 0.0f, 100.0f);
    }

    void InitKernel() override
    {
        const float floatN = static_cast<float>(m_n);
        mMId = m_tuner->AddArgumentScalar(m_m);
        mNId = m_tuner->AddArgumentScalar(m_n);
        mFloatNId = m_tuner->AddArgumentScalar(floatN);
        mDataId = m_tuner->AddArgumentVector(m_data, ktt::ArgumentAccessType::ReadWrite);
        mSymmatId = m_tuner->AddArgumentVector(m_symmat, ktt::ArgumentAccessType::ReadWrite);
        mMeanId = m_tuner->AddArgumentVector(m_mean, ktt::ArgumentAccessType::ReadWrite);

        const ktt::DimensionVector ndRangeDim1D(m_m, 1);
        const ktt::DimensionVector workGroupDim1D(256, 1);
        const ktt::DimensionVector ndRangeDim2D(m_m, m_m);
        const ktt::DimensionVector workGroupDim2D(32, 8);

        m_refMeanDefinition = m_tuner->AddKernelDefinitionFromFile("mean_kernel_reference", m_refKernelFile, ndRangeDim1D, workGroupDim1D);
        m_refReduceDefinition = m_tuner->AddKernelDefinitionFromFile("reduce_kernel_reference", m_refKernelFile, ndRangeDim2D, workGroupDim2D);
        m_refCovarDefinition = m_tuner->AddKernelDefinitionFromFile("covar_kernel_reference", m_refKernelFile, ndRangeDim1D, workGroupDim1D);

        m_meanDefinition = m_tuner->AddKernelDefinitionFromFile("mean_kernel", m_kernelFile, ndRangeDim1D, workGroupDim1D);
        m_reduceDefinition = m_tuner->AddKernelDefinitionFromFile("reduce_kernel", m_kernelFile, ndRangeDim2D, workGroupDim2D);
        m_covarDefinition = m_tuner->AddKernelDefinitionFromFile("covar_kernel", m_kernelFile, ndRangeDim1D, workGroupDim1D);
        m_gemmDefinition = m_tuner->AddKernelDefinitionFromFile("gemm_fast", m_gemmFile, ndRangeDim2D, ktt::DimensionVector());
        m_triangularToSymmetricDefinition = m_tuner->AddKernelDefinitionFromFile("triangular_to_symmetric", m_kernelFile, ndRangeDim2D, workGroupDim2D);

        m_kernel = m_tuner->CreateCompositeKernel("Covariance",
            {m_refMeanDefinition, m_refReduceDefinition, m_refCovarDefinition, m_meanDefinition, m_reduceDefinition,
             m_covarDefinition, m_gemmDefinition, m_triangularToSymmetricDefinition},
            [this](ktt::ComputeInterface& interface)
            {
                const vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
                const uint64_t kernelVariant = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "KERNEL");

                if (kernelVariant == 0)
                {
                    interface.RunKernel(m_refMeanDefinition);
                    interface.RunKernel(m_refReduceDefinition);
                    interface.RunKernel(m_refCovarDefinition);
                }
                else if (kernelVariant == 1)
                {
                    interface.RunKernel(m_meanDefinition);
                    interface.RunKernel(m_reduceDefinition);
                    interface.RunKernel(m_covarDefinition);
                }
                else if (kernelVariant == 2)
                {
                    interface.RunKernel(m_meanDefinition);
                    interface.RunKernel(m_reduceDefinition);
                    interface.RunKernel(m_gemmDefinition);

                    if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SYM_STORE") == 0)
                    {
                        interface.RunKernel(m_triangularToSymmetricDefinition);
                    }
                }
            });

        m_tuner->SetArguments(m_refMeanDefinition, {mMeanId, mDataId, mFloatNId, mMId, mNId});
        m_tuner->SetArguments(m_refReduceDefinition, {mMeanId, mDataId, mMId, mNId});
        m_tuner->SetArguments(m_refCovarDefinition, {mSymmatId, mDataId, mMId, mNId});
        m_tuner->SetArguments(m_meanDefinition, {mMeanId, mDataId, mFloatNId, mMId, mNId});
        m_tuner->SetArguments(m_reduceDefinition, {mMeanId, mDataId, mMId, mNId});
        m_tuner->SetArguments(m_covarDefinition, {mSymmatId, mDataId, mMId, mNId});
        m_tuner->SetArguments(m_gemmDefinition, {mMId, mNId, mDataId, mSymmatId});
        m_tuner->SetArguments(m_triangularToSymmetricDefinition, {mSymmatId, mMId});
    }

    void InitTuningSpace() override
    {
        // Add parameters to tuned kernel
        // Some parameters are commented out to cut down the tuned space - it is now the same as the
        // simpler and commonly tuned space in CLBlast (plus our new parameters).
        // KERNEL: 1 - reference kernels, 0 - edited reference kernels, 2 - use GEMM as third kernel
        m_tuner->AddParameter(m_kernel, "KERNEL", vector<uint64_t>{1, 0, 2});
        m_tuner->AddParameter(m_kernel, "MWG", vector<uint64_t>{16, 32, 64, 128});
        m_tuner->AddParameter(m_kernel, "NWG", vector<uint64_t>{16, 32, 64, 128});
        m_tuner->AddParameter(m_kernel, "KWG", vector<uint64_t>{16, 32});
        m_tuner->AddParameter(m_kernel, "MDIMC", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "NDIMC", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "MDIMA", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "NDIMB", vector<uint64_t>{8, 16, 32});
        m_tuner->AddParameter(m_kernel, "KWI", vector<uint64_t>{2, 8});
        m_tuner->AddParameter(m_kernel, "VWM", vector<uint64_t>{1, 2, 4, 8});
        m_tuner->AddParameter(m_kernel, "VWN", vector<uint64_t>{1, 2, 4, 8});
        m_tuner->AddParameter(m_kernel, "STRM", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "STRN", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "SA", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "SB", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "SYMMETRIC", vector<uint64_t>{0, 1});
        // If SYMMETRIC == 1:
        //   SYM_STORE == 1: if VWM == 1, store the symmetric value right in the GEMM kernel
        //   SYM_STORE == 0: use fourth kernel to make the triangular matrix symmetric
        m_tuner->AddParameter(m_kernel, "SYM_STORE", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "PRECISION", vector<uint64_t>{32});

        auto globalModifier = [](const uint64_t size, const vector<uint64_t>& v) {return (size * v[0] / v[1]);};

        m_tuner->AddThreadModifier(m_kernel, {m_gemmDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"MDIMC", "MWG"}, globalModifier);
        m_tuner->AddThreadModifier(m_kernel, {m_gemmDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"NDIMC", "NWG"}, globalModifier);

        auto localModifier = [](const uint64_t /*size*/, const vector<uint64_t>& v) {return v[0];};
        m_tuner->AddThreadModifier(m_kernel, {m_gemmDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, {"MDIMC"}, localModifier);
        m_tuner->AddThreadModifier(m_kernel, {m_gemmDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, {"NDIMC"}, localModifier);

        auto multipleOfX = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1]); };
        auto multipleOfXMulY = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1] * v[2]); };
        auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v)
        {
            return IsMultiple(v[0], (v[1] * v[2]) / v[3]);
        };

        // Sets constraints: Requirement for unrolling the KWG loop
        m_tuner->AddConstraint(m_kernel, {"KWG", "KWI"}, multipleOfX);

        // Sets constraints: Required for integer MWI and NWI
        m_tuner->AddConstraint(m_kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);

        // Sets constraints: Required for integer MWIA and NWIB
        m_tuner->AddConstraint(m_kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
        m_tuner->AddConstraint(m_kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);

        // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
        m_tuner->AddConstraint(m_kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
        m_tuner->AddConstraint(m_kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);
        // Don't use parameters for polybench reference kernels,
        auto reference = [](const vector<uint64_t>& v)
        {
            if (v[0] == 2)
            {
                return true;
            }
            return v[1] == 32 && v[2] == 32 && v[3] == 32 && v[4] == 8 && v[5] == 8 && v[6] == 8 && v[7] == 8 && v[8] == 2
                && v[9] == 1 && v[10] == 1 && v[11] == 0 && v[12] == 0 && v[13] == 1 && v[14] == 1 && v[15] == 0;
        };

        m_tuner->AddConstraint(m_kernel, {"KERNEL", "MWG", "NWG", "KWG", "MDIMC", "NDIMC", "MDIMA", "NDIMB", "KWI", "VWM", "VWN", "STRM",
            "STRN", "SA", "SB", "SYMMETRIC"}, reference);

        // New NVidia GPUs have max. workgroup size
        auto maxWgSize = [this](const vector<uint64_t>& v) {return v[0] * v[1] <= MAX_WORK_GROUP_SIZE;};
        m_tuner->AddConstraint(m_kernel, {"MDIMC", "NDIMC"}, maxWgSize);

        // Symmetric store can't be used for vectors
        auto symmetric = [](const vector<uint64_t>& v)
        {
            if (v[0] == 1)
            {
                if (v[1] == 1)
                {
                    return v[2] == 1;
                }
                return true;
            }
            return v[1] == 0;
        };

        m_tuner->AddConstraint(m_kernel, {"SYMMETRIC", "SYM_STORE", "VWM"}, symmetric);
    }

    void InitReference() override
    {
        m_tuner->SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 128.0);

        m_tuner->SetReferenceComputation(mSymmatId, [this](void* buffer)
        {
            const float floatN = static_cast<float>(m_n);
            vector<float> mean(m_m, 0.0f);
            float* symmat = static_cast<float*>(buffer);

            for (int j = 0; j < m_m; ++j)
            {
                mean[j] = 0.0;
                for (int i = 0; i < m_n; ++i)
                {
                    mean[j] += m_data[i * m_m + j];
                }
                mean[j] /= floatN;
            }

            for (int i = 0; i < m_n; ++i)
            {
                for (int j = 0; j < m_m; ++j)
                {
                    m_data[i * m_m + j] -= mean[j];
                }
            }

            for (int j1 = 0; j1 < m_m; ++j1)
            {
                for (int j2 = j1; j2 < m_m; ++j2)
                {
                    symmat[j1 * m_m + j2] = 0.0;
                    for (int i = 0; i < m_n; ++i)
                    {
                        symmat[j1 * m_m + j2] += m_data[i * m_m + j1] * m_data[i * m_m + j2];
                    }
                    symmat[j2 * m_m + j1] = symmat[j1 * m_m + j2];
                }
            }
        });

        m_tuner->SetReferenceComputation(mMeanId, [this](void* buffer)
        {
            const float floatN = static_cast<float>(m_n);
            float* mean = static_cast<float*>(buffer);

            for (int j = 0; j < m_m; ++j)
            {
                mean[j] = 0.0;
                for (int i = 0; i < m_n; ++i)
                {
                    mean[j] += m_data[i * m_m + j];
                }
                mean[j] /= floatN;
            }
        });
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Covariance> covariance = Covariance::Create<Covariance>(argc, argv, 1, "Examples/Covariance", "Covariance");
    covariance->Run();

    return 0;
}
