#include "../ExampleReferenceKernel.h"
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace std;

class Nbody: public ExampleReferenceKernel {
protected:
    Nbody(int argc, char **argv,
              string exampleFolderPath, string defaultKernelFileBaseName,
              string defaultReferenceKernelFileBaseName):
        ExampleReferenceKernel(argc, argv, exampleFolderPath, defaultKernelFileBaseName,
                defaultReferenceKernelFileBaseName)
    {
        // The total number of computations grows quadratically with problem size, hence the sqrt.
        m_inputSize = ktt::DimensionVector(128 * 1024);
    }

    friend ExampleReferenceKernel;

    // Declare and initialize data
    ktt::DimensionVector m_inputSize;
    int m_numberOfBodies;

    const float timeDelta = 0.001f;
    const float damping = 0.5f;
    const float softeningSqr = 0.1f * 0.1f;

    std::vector<float> m_oldBodyInfo;
    std::vector<float> m_oldPosX, m_oldPosY, m_oldPosZ;
    std::vector<float> m_bodyMass;

    std::vector<float> m_newBodyInfo;

    std::vector<float> m_oldBodyVel;
    std::vector<float> m_newBodyVel;
    std::vector<float> m_oldVelX, m_oldVelY, m_oldVelZ;

    ktt::ArgumentId m_oldBodyInfoId;
    ktt::ArgumentId m_oldPosXId, m_oldPosYId, m_oldPosZId;
    ktt::ArgumentId m_massId;
    ktt::ArgumentId m_newBodyInfoId;

    ktt::ArgumentId m_oldVelId;
    ktt::ArgumentId m_oldVelXId, m_oldVelYId, m_oldVelZId;
    ktt::ArgumentId m_newBodyVelId;

    ktt::ArgumentId m_deltaTimeId, m_dampingId, m_softeningSqrId;
    ktt::ArgumentId m_numberOfBodiesId;

    void InitCLI() override
    {
        ExampleBase::InitCLI();
        UseInputSizeOption(1, m_inputSize);
    }

    void InitData() override
    {
        m_numberOfBodies = m_inputSize.GetSizeX();
        // Total NDRange size matches number of grid points
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

        // Initialize data
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, 20.0f);

        FillBuffers({&m_oldPosX, &m_oldPosY, &m_oldPosZ, &m_bodyMass, &m_oldVelX, &m_oldVelY, &m_oldVelZ},
        0.f, 20.f);

        for (int i = 0; i < m_numberOfBodies; ++i)
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

        m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);

        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner->SetCompilerOptions("-cl-fast-relaxed-math");
        }
        else
        {
            m_tuner->SetCompilerOptions("-use_fast_math");
        }

    }

    void InitKernel() override
    {
        const ktt::DimensionVector ndRangeDimensions(m_numberOfBodies, 1);
        const ktt::DimensionVector workGroupDimensions(1, 1);

        // Add all arguments utilized by m_kernels
        m_oldBodyInfoId = m_tuner->AddArgumentVector(m_oldBodyInfo, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosXId = m_tuner->AddArgumentVector(m_oldPosX, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosYId = m_tuner->AddArgumentVector(m_oldPosY, ktt::ArgumentAccessType::ReadOnly);
        m_oldPosZId = m_tuner->AddArgumentVector(m_oldPosZ, ktt::ArgumentAccessType::ReadOnly);
        m_massId = m_tuner->AddArgumentVector(m_bodyMass, ktt::ArgumentAccessType::ReadOnly);
        m_newBodyInfoId = m_tuner->AddArgumentVector(m_newBodyInfo, ktt::ArgumentAccessType::WriteOnly);

        m_oldVelId = m_tuner->AddArgumentVector(m_oldBodyVel, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelXId = m_tuner->AddArgumentVector(m_oldVelX, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelYId = m_tuner->AddArgumentVector(m_oldVelY, ktt::ArgumentAccessType::ReadOnly);
        m_oldVelZId = m_tuner->AddArgumentVector(m_oldVelZ, ktt::ArgumentAccessType::ReadOnly);
        m_newBodyVelId = m_tuner->AddArgumentVector(m_newBodyVel, ktt::ArgumentAccessType::WriteOnly);

        m_deltaTimeId = m_tuner->AddArgumentScalar(timeDelta);
        m_dampingId = m_tuner->AddArgumentScalar(damping);
        m_softeningSqrId = m_tuner->AddArgumentScalar(softeningSqr);
        m_numberOfBodiesId = m_tuner->AddArgumentScalar(m_numberOfBodies);

        InitKernelDefault("nbody_kernel", "Nbody", ndRangeDimensions, {m_deltaTimeId,
            m_oldBodyInfoId, m_oldPosXId, m_oldPosYId, m_oldPosZId, m_massId, m_newBodyInfoId, // position
            m_oldVelId, m_oldVelXId, m_oldVelYId, m_oldVelZId, m_newBodyVelId, // velocity
            m_dampingId, m_softeningSqrId, m_numberOfBodiesId});

    }

    void InitTuningSpace() override
    {
        // Multiply work-group size in dimensions x and y by two parameters that follow (effectively setting work-group size to parameters' values)
        m_tuner->AddParameter(m_kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);

        m_tuner->AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16});
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);

        m_tuner->AddParameter(m_kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8});

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR",
            ktt::ModifierAction::Divide);

        m_tuner->AddParameter(m_kernel, "INNER_UNROLL_FACTOR1", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        m_tuner->AddParameter(m_kernel, "INNER_UNROLL_FACTOR2", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
        // m_tuner->AddParameter(m_kernel, "INNER_UNROLL_FACTOR1", std::vector<uint64_t>{0});
        // m_tuner->AddParameter(m_kernel, "INNER_UNROLL_FACTOR2", std::vector<uint64_t>{0});


        m_tuner->AddParameter(m_kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});

        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner->AddParameter(m_kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
            m_tuner->AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8, 16});
        }
        else
        {
            m_tuner->AddParameter(m_kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
            m_tuner->AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
        }

        // Add conditions
        auto lteq = [](const std::vector<uint64_t>& vector) {return vector.at(0) <= vector.at(1);};
        m_tuner->AddConstraint(m_kernel, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, lteq);
        auto lteq256 = [](const std::vector<uint64_t>& vector) {return vector.at(0) * vector.at(1) <= 256;};
        m_tuner->AddConstraint(m_kernel, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, lteq256);
        auto vectorizedSoA = [](const std::vector<uint64_t>& vector) {return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1);};
        m_tuner->AddConstraint(m_kernel, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);
        auto blockSize = [](const std::vector<uint64_t>& vector) {return (vector.at(0) * vector.at(1) >= 64) && (vector.at(0) * vector.at(1) <= 1024);};
        m_tuner->AddConstraint(m_kernel, std::vector<std::string>{"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, blockSize);
    }

    void InitReference() override
    {
        const ktt::DimensionVector refNdRangeDimensions(m_numberOfBodies, 1);
        const ktt::DimensionVector refWorkGroupDimensions(64);
        InitReferenceKernelDefault("nbody_kernel_reference", refNdRangeDimensions, refWorkGroupDimensions, 
            {m_deltaTimeId, m_oldBodyInfoId, m_newBodyInfoId, m_oldVelId, m_newBodyVelId,
            m_dampingId, m_softeningSqrId}, {m_newBodyVelId, m_newBodyInfoId}, 0.001);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Nbody> nbody = Nbody::Create<Nbody>(
        argc, argv, "Examples/Nbody", "Nbody", "NbodyReference"
    );
    nbody->Run();

    return 0;
}