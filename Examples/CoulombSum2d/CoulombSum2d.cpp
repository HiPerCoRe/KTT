#include "../ExampleReferenceKernel.h"
#include <memory>

using namespace std;

class CoulombSum2d : public ExampleReferenceKernel {
protected:
    CoulombSum2d(int argc, char **argv, int defaultProblemSize, string exampleFolderPath,
                 string defaultKernelFileBaseName, string defaultRefKernelFileBaseName) :
        ExampleReferenceKernel(argc, argv, defaultProblemSize, exampleFolderPath,
                               defaultKernelFileBaseName, defaultRefKernelFileBaseName),
        // Since CoulombSum2d has O(n²) complexity (gridPoints × atoms), scale grid dimensions
        // with the fourth root of problem size to keep total work proportional
        m_gridWidth(static_cast<size_t>(sqrt(m_problemSize)) * 16),
        m_gridHeight(static_cast<size_t>(sqrt(m_problemSize)) * 16),
        m_ndRangeDimensions(m_gridWidth, m_gridHeight),
        m_numberOfAtoms(4000)
    {
    }

    friend ExampleReferenceKernel;

    size_t m_gridWidth;
    size_t m_gridHeight;

    // Total NDRange size matches number of grid points
    const ktt::DimensionVector m_ndRangeDimensions;
    const ktt::DimensionVector m_workGroupDimensions{1, 1};

    const float m_gridSpacing = 0.5f;
    const int m_numberOfAtoms;

    vector<float> m_atomInfo;
    vector<float> m_atomInfoX;
    vector<float> m_atomInfoY;
    vector<float> m_atomInfoZ;
    vector<float> m_atomInfoW;
    vector<float> m_energyGrid;

    ktt::ArgumentId m_atomInfoId;
    ktt::ArgumentId m_atomInfoXId;
    ktt::ArgumentId m_atomInfoYId;
    ktt::ArgumentId m_atomInfoZId;
    ktt::ArgumentId m_atomInfoWId;
    ktt::ArgumentId m_numberOfAtomsId;
    ktt::ArgumentId m_gridSpacingId;
    ktt::ArgumentId m_energyGridId;

    void InitData() override
    {
        // Declare data variables
        const size_t numberOfGridPoints = m_gridWidth * m_gridHeight;
        m_atomInfo.resize(4 * m_numberOfAtoms);
        m_atomInfoX.resize(m_numberOfAtoms);
        m_atomInfoY.resize(m_numberOfAtoms);
        m_atomInfoZ.resize(m_numberOfAtoms);
        m_atomInfoW.resize(m_numberOfAtoms);
        m_energyGrid.resize(numberOfGridPoints, 0.0f);

        FillBuffers<float>({&m_atomInfoX, &m_atomInfoY, &m_atomInfoZ}, 0.0f, 40.0f);
        FillBuffers<float>({&m_atomInfoW}, 0.0f, 1.0f);

        for (size_t i = 0; i < static_cast<size_t>(m_numberOfAtoms); ++i)
        {
            m_atomInfo[4 * i] = m_atomInfoX[i];
            m_atomInfo[4 * i + 1] = m_atomInfoY[i];
            m_atomInfo[4 * i + 2] = m_atomInfoZ[i];
            m_atomInfo[4 * i + 3] = m_atomInfoW[i];
        }
    }

    void InitKernel() override
    {
        // Add all kernel arguments
        m_atomInfoId = m_tuner->AddArgumentVector(m_atomInfo, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoXId = m_tuner->AddArgumentVector(m_atomInfoX, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoYId = m_tuner->AddArgumentVector(m_atomInfoY, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoZId = m_tuner->AddArgumentVector(m_atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoWId = m_tuner->AddArgumentVector(m_atomInfoW, ktt::ArgumentAccessType::ReadOnly);
        m_numberOfAtomsId = m_tuner->AddArgumentScalar(m_numberOfAtoms);
        m_gridSpacingId = m_tuner->AddArgumentScalar(m_gridSpacing);
        m_energyGridId = m_tuner->AddArgumentVector(m_energyGrid, ktt::ArgumentAccessType::ReadWrite);

        // Configure main kernel
        InitKernelDefault("directCoulombSum", "CoulombSum", m_ndRangeDimensions,
            {m_atomInfoId, m_atomInfoXId, m_atomInfoYId, m_atomInfoZId, m_atomInfoWId,
             m_numberOfAtomsId, m_gridSpacingId, m_energyGridId});
    }

    void InitTuningSpace() override
    {
        UseFastMath();

        m_tuner->AddParameter(m_kernel, "INNER_UNROLL_FACTOR", vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        m_tuner->AddParameter(m_kernel, "USE_CONSTANT_MEMORY", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "VECTOR_TYPE", vector<uint64_t>{1, 2, 4, 8});
        m_tuner->AddParameter(m_kernel, "USE_SOA", vector<uint64_t>{0, 1, 2});

        // Using vectorized SoA only makes sense when vectors are longer than 1.
        auto vectorizedSoA = [](const vector<uint64_t>& vector) {return vector[0] > 1 || vector[1] != 2;};
        m_tuner->AddConstraint(m_kernel, {"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);

        // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR.
        m_tuner->AddParameter(m_kernel, "OUTER_UNROLL_FACTOR", vector<uint64_t>{1, 2, 4, 8});
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
            ktt::ModifierAction::Divide);

        // Multiply work-group size in dimensions x and y by the following parameters (effectively setting work-group size to their values).
        m_tuner->AddParameter(m_kernel, "WORK_GROUP_SIZE_X", vector<uint64_t>{4, 8, 16, 32});
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Divide);

        m_tuner->AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", vector<uint64_t>{1, 2, 4, 8, 16, 32});
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Divide);
    }

    void InitReference() override
    {
        const ktt::DimensionVector referenceWorkGroupDimensions{16, 16};
        const ktt::DimensionVector referenceNdRangeDimensions{m_ndRangeDimensions.GetSizeX()/referenceWorkGroupDimensions.GetSizeX(),
                                                              m_ndRangeDimensions.GetSizeY()/referenceWorkGroupDimensions.GetSizeY()};
        // Configure reference kernel
        InitReferenceKernelDefault("directCoulombSumReference", referenceNdRangeDimensions, referenceWorkGroupDimensions,
            {m_atomInfoId, m_numberOfAtomsId, m_gridSpacingId, m_energyGridId},
            {m_energyGridId}, 0.01);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<CoulombSum2d> coulombSum2d = CoulombSum2d::Create<CoulombSum2d>(argc, argv, 256, "Examples/CoulombSum2d",
        "CoulombSum2d", "CoulombSum2dReference");
    coulombSum2d->Run();

    return 0;
}
