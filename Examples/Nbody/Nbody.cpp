#include "../ExampleReferenceKernel.h"
#include <memory>

using namespace std;

class Nbody : public ExampleReferenceKernel {
protected:
    Nbody(std::shared_ptr<ExampleRefKernelConfiguration> config, int defaultProblemSize, string exampleFolderPath,
          string defaultKernelFileBaseName, string defaultRefKernelFileBaseName) :
        ExampleReferenceKernel(config, defaultProblemSize, exampleFolderPath,
                               defaultKernelFileBaseName, defaultRefKernelFileBaseName),
        m_numberOfBodies(static_cast<size_t>(sqrt(m_problemSize)) * 1024),
        m_ndRangeDimensions(m_numberOfBodies, 1)
    {
    }

    friend ExampleReferenceKernel;

    size_t m_numberOfBodies;

    // Total NDRange size matches number of grid points
    const ktt::DimensionVector m_ndRangeDimensions;
    const ktt::DimensionVector m_workGroupDimensions{1, 1};
    const ktt::DimensionVector m_referenceWorkGroupDimensions{64};

    const float m_timeDelta = 0.001f;
    const float m_damping = 0.5f;
    const float m_softeningSqr = 0.1f * 0.1f;

    vector<float> m_oldBodyInfo;
    vector<float> m_oldPosX, m_oldPosY, m_oldPosZ;
    vector<float> m_bodyMass;

    vector<float> m_newBodyInfo;

    vector<float> m_oldBodyVel;
    vector<float> m_newBodyVel;
    vector<float> m_oldVelX;
    vector<float> m_oldVelY;
    vector<float> m_oldVelZ;

    ktt::ArgumentId m_oldBodyInfoId;
    ktt::ArgumentId m_oldPosXId;
    ktt::ArgumentId m_oldPosYId;
    ktt::ArgumentId m_oldPosZId;
    ktt::ArgumentId m_massId;
    ktt::ArgumentId m_newBodyInfoId;

    ktt::ArgumentId m_oldVelId;
    ktt::ArgumentId m_oldVelXId;
    ktt::ArgumentId m_oldVelYId;
    ktt::ArgumentId m_oldVelZId;
    ktt::ArgumentId m_newBodyVelId;

    ktt::ArgumentId m_deltaTimeId;
    ktt::ArgumentId m_dampingId;
    ktt::ArgumentId m_softeningSqrId;
    ktt::ArgumentId m_numberOfBodiesId;

    void InitData() override
    {
        // Declare data variables
        m_oldBodyInfo.resize(4 * m_numberOfBodies);
        m_oldPosX.resize(m_numberOfBodies);
        m_oldPosY.resize(m_numberOfBodies);
        m_oldPosZ.resize(m_numberOfBodies);
        m_bodyMass.resize(m_numberOfBodies);

        m_newBodyInfo.resize(4 * m_numberOfBodies, 0.f);

        m_oldBodyVel.resize(4 * m_numberOfBodies);
        m_newBodyVel.resize(4 * m_numberOfBodies);
        m_oldVelX.resize(m_numberOfBodies);
        m_oldVelY.resize(m_numberOfBodies);
        m_oldVelZ.resize(m_numberOfBodies);

        FillBuffers<float>({&m_oldPosX, &m_oldPosY, &m_oldPosZ, &m_bodyMass,
                            &m_oldVelX, &m_oldVelY, &m_oldVelZ}, 0.0f, 20.0f);

        for (size_t i = 0; i < m_numberOfBodies; ++i)
        {
            m_oldBodyInfo[4 * i] = m_oldPosX[i];
            m_oldBodyInfo[4 * i + 1] = m_oldPosY[i];
            m_oldBodyInfo[4 * i + 2] = m_oldPosZ[i];
            m_oldBodyInfo[4 * i + 3] = m_bodyMass[i];

            m_oldBodyVel[4 * i] = m_oldVelX[i];
            m_oldBodyVel[4 * i + 1] = m_oldVelY[i];
            m_oldBodyVel[4 * i + 2] = m_oldVelZ[i];
            m_oldBodyVel[4 * i + 3] = 0.0f;
        }
    }

    void InitKernel() override
    {
        // Add all arguments utilized by kernels
        m_oldBodyInfoId = m_tuner.AddArgumentVector(m_oldBodyInfo, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosXId = m_tuner.AddArgumentVector(m_oldPosX, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosYId = m_tuner.AddArgumentVector(m_oldPosY, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosZId = m_tuner.AddArgumentVector(m_oldPosZ, ktt::ArgumentAccessType::ReadOnly);
        m_massId = m_tuner.AddArgumentVector(m_bodyMass, ktt::ArgumentAccessType::ReadOnly);
        m_newBodyInfoId = m_tuner.AddArgumentVector(m_newBodyInfo, ktt::ArgumentAccessType::WriteOnly);

        m_oldVelId = m_tuner.AddArgumentVector(m_oldBodyVel, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelXId = m_tuner.AddArgumentVector(m_oldVelX, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelYId = m_tuner.AddArgumentVector(m_oldVelY, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelZId = m_tuner.AddArgumentVector(m_oldVelZ, ktt::ArgumentAccessType::ReadOnly);
        m_newBodyVelId = m_tuner.AddArgumentVector(m_newBodyVel, ktt::ArgumentAccessType::WriteOnly);

        m_deltaTimeId = m_tuner.AddArgumentScalar(m_timeDelta);
        m_dampingId = m_tuner.AddArgumentScalar(m_damping);
        m_softeningSqrId = m_tuner.AddArgumentScalar(m_softeningSqr);
        m_numberOfBodiesId = m_tuner.AddArgumentScalar(m_numberOfBodies);

        // Configure main kernel
        InitKernelDefault("nbody_kernel", "Nbody", m_ndRangeDimensions,
            {m_deltaTimeId,
             m_oldBodyInfoId, m_oldPosXId, m_oldPosYId, m_oldPosZId, m_massId, m_newBodyInfoId, // position
             m_oldVelId, m_oldVelXId, m_oldVelYId, m_oldVelZId, m_newBodyVelId, // velocity
             m_dampingId, m_softeningSqrId, m_numberOfBodiesId});
    }

    void InitTuningSpace() override
    {
        UseFastMath();

        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_X", vector<uint64_t>{16, 32, 64, 128, 256, 512});

        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Divide);

        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", vector<uint64_t>{1, 2, 4, 8, 16});
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);

        m_tuner.AddParameter(m_kernel, "OUTER_UNROLL_FACTOR", vector<uint64_t>{1, 2, 4, 8});

        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
            ktt::ModifierAction::Divide);

        m_tuner.AddParameter(m_kernel, "INNER_UNROLL_FACTOR1", vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        m_tuner.AddParameter(m_kernel, "INNER_UNROLL_FACTOR2", vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});

        m_tuner.AddParameter(m_kernel, "USE_SOA", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "LOCAL_MEM", vector<uint64_t>{0, 1});

        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner.AddParameter(m_kernel, "USE_CONSTANT_MEMORY", vector<uint64_t>{0});
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", vector<uint64_t>{1, 2, 4, 8, 16});
        }
        else
        {
            m_tuner.AddParameter(m_kernel, "USE_CONSTANT_MEMORY", vector<uint64_t>{0});
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", vector<uint64_t>{1, 2, 4});
        }

        // Add conditions
        auto lteq = [](const vector<uint64_t>& vector) {return vector.at(0) <= vector.at(1);};
        m_tuner.AddConstraint(m_kernel, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, lteq);
        auto lteq256 = [](const vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) <= 256;};
        m_tuner.AddConstraint(m_kernel, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, lteq256);
        auto vectorizedSoA = [](const vector<uint64_t>& vector) {return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1);};
        m_tuner.AddConstraint(m_kernel, {"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);
        auto blockSize = [](const vector<uint64_t>& vector) {return (vector.at(0) * vector.at(1) >= 64) && (vector.at(0) * vector.at(1) <= 1024);};
        m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, blockSize);
    }

    void InitReference() override
    {
        // Configure reference kernel
        InitReferenceKernelDefault("nbody_kernel_reference", m_ndRangeDimensions, m_referenceWorkGroupDimensions,
            {m_deltaTimeId, m_oldBodyInfoId, m_newBodyInfoId, m_oldVelId, m_newBodyVelId, m_dampingId, m_softeningSqrId},
            {m_newBodyInfoId, m_newBodyVelId}, 0.001);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Nbody> nbody = Nbody::Create<Nbody>(argc, argv, 128*128, "Examples/Nbody", "Nbody", "NbodyReference");
    nbody->Run();

    return 0;
}
