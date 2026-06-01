#include "../ExampleReferenceComputation.h"
#include <memory>
#include <iostream>

using namespace std;

class Bicg : public ExampleReferenceComputation {
protected:
    Bicg(shared_ptr<ExampleConfiguration> config, int defaultProblemSize, string exampleFolderPath,
         string defaultKernelFileBaseName) :
        ExampleReferenceComputation(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName),
        // Bicg has O(m × n) complexity. For square matrices where m = n,
        // we scale with square root of problem size to keep total work proportional
        m_m(static_cast<int>(sqrt(m_problemSize)) * 1024),
        m_n(static_cast<int>(sqrt(m_problemSize)) * 1024)
    {
    }

    friend ExampleBase;

    int m_m;
    int m_n;

    const int WORK_GROUP_X = 256;
    const int WORK_GROUP_Y = 1;

    // New NVidia GPUs have max. work-group size of 1024
    const uint64_t MAX_WORK_GROUP_SIZE = 1024;

    vector<float> m_A;
    vector<float> m_x1;
    vector<float> m_x2;
    vector<float> m_y1;
    vector<float> m_y2;

    ktt::ArgumentId m_AId;
    ktt::ArgumentId m_x1Id;
    ktt::ArgumentId m_x2Id;
    ktt::ArgumentId m_y1Id;
    ktt::ArgumentId m_y2Id;
    ktt::ArgumentId m_mId;
    ktt::ArgumentId m_nId;

    ktt::KernelDefinitionId m_definitionFused;
    ktt::KernelDefinitionId m_definitionReduction1;
    ktt::KernelDefinitionId m_definitionReduction2;

    void InitData() override
    {
        m_A.resize(m_n * m_m);
        m_x1.resize(m_m);
        m_x2.resize(m_n);
        m_y1.resize(m_n * m_m / 16, 0.0f);
        m_y2.resize(m_m * m_n / 128, 0.0f);

        FillBuffers<float>({&m_A, &m_x1, &m_x2}, 0.0f, 100.0f);
    }

    void InitKernel() override
    {
        m_mId = m_tuner.AddArgumentScalar(m_m);
        m_nId = m_tuner.AddArgumentScalar(m_n);
        m_AId = m_tuner.AddArgumentVector(m_A, ktt::ArgumentAccessType::ReadWrite);
        m_x1Id = m_tuner.AddArgumentVector(m_x1, ktt::ArgumentAccessType::ReadOnly);
        m_x2Id = m_tuner.AddArgumentVector(m_x2, ktt::ArgumentAccessType::ReadOnly);
        m_y1Id = m_tuner.AddArgumentVector(m_y1, ktt::ArgumentAccessType::ReadWrite);
        m_y2Id = m_tuner.AddArgumentVector(m_y2, ktt::ArgumentAccessType::ReadWrite);

        const ktt::DimensionVector ndRangeDimensions(m_m, m_n / 64);
        const ktt::DimensionVector workGroupDimensions(32, 4);
        const ktt::DimensionVector referenceNdRangeDimensions1(static_cast<size_t>(ceil(m_n * 1. / WORK_GROUP_X)), 1);
        const ktt::DimensionVector referenceNdRangeDimensions2(static_cast<size_t>(ceil(m_m * 1. / WORK_GROUP_X)), 1);
        const ktt::DimensionVector referenceWorkGroupDimensions(WORK_GROUP_X, WORK_GROUP_Y);

        m_definitionFused = m_tuner.AddKernelDefinitionFromFile("bicgFused", m_kernelFile, ndRangeDimensions, workGroupDimensions);
        m_definitionReduction1 = m_tuner.AddKernelDefinitionFromFile("bicgReduction1", m_kernelFile, referenceNdRangeDimensions1, referenceWorkGroupDimensions);
        m_definitionReduction2 = m_tuner.AddKernelDefinitionFromFile("bicgReduction2", m_kernelFile, referenceNdRangeDimensions1, referenceWorkGroupDimensions);

        m_kernel = m_tuner.CreateCompositeKernel("BicgPolyBenchAndFused", {m_definitionFused, m_definitionReduction1, m_definitionReduction2},
            [this](ktt::ComputeInterface& interface)
            {
                const vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
                if (!m_config->useProfiling)
                {
                    interface.RunKernel(m_definitionFused);
                }
                else
                {
                    interface.RunKernelWithProfiling(m_definitionFused);
                }

                if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "ATOMICS") == 0)
                {
                    interface.RunKernel(m_definitionReduction1);
                    interface.RunKernel(m_definitionReduction2);
                }
            });

        m_tuner.SetArguments(m_definitionFused, {m_AId, m_x1Id, m_y1Id, m_x2Id, m_y2Id, m_mId, m_nId});
        m_tuner.SetArguments(m_definitionReduction1, {m_mId, m_nId, m_y1Id});
        m_tuner.SetArguments(m_definitionReduction2, {m_mId, m_nId, m_y2Id});
    }

    void InitTuningSpace() override
    {
        m_tuner.AddParameter(m_kernel, "FUSED", vector<uint64_t>{2});
        m_tuner.AddParameter(m_kernel, "BICG_BATCH", vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
        m_tuner.AddParameter(m_kernel, "USE_SHARED_MATRIX", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "USE_SHARED_VECTOR_1", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "USE_SHARED_VECTOR_2", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "USE_SHARED_REDUCTION_1", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "USE_SHARED_REDUCTION_2", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "ATOMICS", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "UNROLL_BICG_STEP", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "ROWS_PROCESSED", vector<uint64_t>{128, 256, 512, 1024});
        m_tuner.AddParameter(m_kernel, "TILE", vector<uint64_t>{16, 32, 64});

        auto globalModifierX = [m = m_m](const uint64_t, const vector<uint64_t>& v) {return m / v.at(0);};
        auto globalModifierY = [n = m_n](const uint64_t, const vector<uint64_t>& v) {return n / v.at(0);};
        auto localModifierX = [](const uint64_t, const vector<uint64_t>& v) {return v.at(0);};
        auto localModifierY = [](const uint64_t, const vector<uint64_t>& v) {return v.at(0) / v.at(1);};

        m_tuner.AddThreadModifier(m_kernel, {m_definitionFused}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"TILE"}, globalModifierX);
        m_tuner.AddThreadModifier(m_kernel, {m_definitionFused}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"ROWS_PROCESSED"}, globalModifierY);
        m_tuner.AddThreadModifier(m_kernel, {m_definitionFused}, ktt::ModifierType::Local, ktt::ModifierDimension::X, {"TILE"}, localModifierX);
        m_tuner.AddThreadModifier(m_kernel, {m_definitionFused}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, {"TILE", "BICG_BATCH"}, localModifierY);

        auto fused = [](const vector<uint64_t>& v) {return v.at(0) == 2 || ((v.at(0) == 0 || v.at(0) == 1) && v.at(1) == 4 && v.at(2) == 1 && v.at(3) == 1 && v.at(4) == 1 && v.at(5) == 1 && v.at(6) == 1 && v.at(7) == 1 && v.at(8) == 1 && v.at(9) == 512 && v.at(10) == 32); };
        m_tuner.AddConstraint(m_kernel, {"FUSED", "BICG_BATCH", "USE_SHARED_MATRIX", "USE_SHARED_VECTOR_1", "USE_SHARED_VECTOR_2", "USE_SHARED_REDUCTION_1", "USE_SHARED_REDUCTION_2", "ATOMICS", "UNROLL_BICG_STEP", "ROWS_PROCESSED", "TILE"}, fused);

        auto maxWgSize = [this](const vector<uint64_t>& v) {return (v.at(0) * v.at(0) / v.at(1) <= MAX_WORK_GROUP_SIZE) && (v.at(1) <= v.at(0)); };
        m_tuner.AddConstraint(m_kernel, {"TILE", "BICG_BATCH"}, maxWgSize);
    }

    void InitReference() override
    {
        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideRelativeComparison, 0.001);
        m_tuner.SetValidationRange(m_y1Id, m_n);
        m_tuner.SetValidationRange(m_y2Id, m_m);

        m_tuner.SetReferenceComputation(m_y1Id, [this](void* buffer)
        {
            float* y1 = static_cast<float*>(buffer);

            for (int i = 0; i < m_n; ++i)
            {
                y1[i] = 0.0f;
                for (int j = 0; j < m_m; ++j)
                {
                    y1[i] = y1[i] + m_A[i * m_m + j] * m_x1[j];
                }
            }
        });

        m_tuner.SetReferenceComputation(m_y2Id, [this](void* buffer)
        {
            float* y2 = static_cast<float*>(buffer);

            for (int i = 0; i < m_m; ++i)
            {
                y2[i] = 0.0f;
            }

            for (int i = 0; i < m_n; ++i)
            {
                for (int j = 0; j < m_m; ++j)
                {
                    y2[j] = y2[j] + m_x2[i] * m_A[i * m_m + j];
                }
            }
        });
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Bicg> bicg = Bicg::Create<Bicg>(argc, argv, 256, "Examples/Bicg", "Bicg");
    bicg->Run();

    return 0;
}
