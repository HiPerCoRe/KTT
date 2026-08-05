/******************************************************************************
 * This is a dummy example -- it uses fairy efficient version of 3D Coulomb sum
 * but does not tune anything.
 * The reason of existence of this example is testing stability of time and
 * power measurement.
 */

#include "../ExampleBase.h"
#include <cstdint>
#include <memory>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

using namespace std;

class Dummy : public ExampleBase {
protected:
    Dummy(shared_ptr<ExampleConfiguration> config, int defaultProblemSize,
          string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        m_gridSize = 256;
        m_atoms = 1024;

        m_tuner->SetTimeUnit(ktt::TimeUnit::Microseconds);
        UseFastMath();

        // Set precise measurement by default.
        if (m_config->preciseParams == nullopt) {
            m_config->preciseParams = ktt::PreciseMeasurementParameters(2000, 20000, 0.005, ktt::DurationCalculationMethod::Minimum);
        }
    }

    friend ExampleBase;

    // Sleep in the manipulator (can be randomized to 0, sleepDuration)
    // (makes power measurement more challenging due to changes in GPU temperature)
    const unsigned int sleepDuration = 0;
    const bool randomizeSleep = true;

    int m_gridSize;
    int m_atoms;

    vector<float> m_atomInfoX;
    vector<float> m_atomInfoY;
    vector<float> m_atomInfoZ;
    vector<float> m_atomInfoW;
    vector<float> m_atomInfo;
    vector<float> m_energyGrid;

    ktt::ArgumentId m_atomInfoId;
    ktt::ArgumentId m_atomInfoXId;
    ktt::ArgumentId m_atomInfoYId;
    ktt::ArgumentId m_atomInfoZId;
    ktt::ArgumentId m_atomInfoWId;
    ktt::ArgumentId m_atomsId;
    ktt::ArgumentId m_gridSpacingId;
    ktt::ArgumentId m_gridDimId;
    ktt::ArgumentId m_energyGridId;

    float m_gridSpacing = 0.5f;

    void InitData() override
    {
        m_atomInfoX.resize(m_atoms);
        m_atomInfoY.resize(m_atoms);
        m_atomInfoZ.resize(m_atoms);
        m_atomInfoW.resize(m_atoms);
        m_atomInfo.resize(4 * m_atoms);
        m_energyGrid.resize(m_gridSize * m_gridSize * m_gridSize, 0.0f);

        // Initialize atom position and charge data
        FillBuffers<float>({&m_atomInfoX, &m_atomInfoY, &m_atomInfoZ}, 0.0f, 20.0f);
        FillBuffers<float>({&m_atomInfoW}, 0.0f, 0.5f);

        for (int i = 0; i < m_atoms; ++i)
        {
            m_atomInfo[4 * i] = m_atomInfoX[i];
            m_atomInfo[4 * i + 1] = m_atomInfoY[i];
            m_atomInfo[4 * i + 2] = m_atomInfoZ[i];
            m_atomInfo[4 * i + 3] = m_atomInfoW[i];
        }
    }

    void InitKernel() override
    {
        const ktt::DimensionVector ndRangeDimensions(m_gridSize / 32, m_gridSize / 4, m_gridSize);
        const ktt::DimensionVector workGroupDimensions(32, 4);

        m_atomInfoId = m_tuner->AddArgumentVector(m_atomInfo, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoXId = m_tuner->AddArgumentVector(m_atomInfoX, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoYId = m_tuner->AddArgumentVector(m_atomInfoY, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoZId = m_tuner->AddArgumentVector(m_atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
        m_atomInfoWId = m_tuner->AddArgumentVector(m_atomInfoW, ktt::ArgumentAccessType::ReadOnly);
        m_atomsId = m_tuner->AddArgumentScalar(m_atoms);
        m_gridSpacingId = m_tuner->AddArgumentScalar(m_gridSpacing);
        m_gridDimId = m_tuner->AddArgumentScalar(m_gridSize);
        m_energyGridId = m_tuner->AddArgumentVector(m_energyGrid, ktt::ArgumentAccessType::WriteOnly);

        InitKernelDefault("directCoulombSum", "CoulombSum", ndRangeDimensions,
                          {m_atomInfoId, m_atomInfoXId, m_atomInfoYId, m_atomInfoZId, m_atomInfoWId,
                           m_atomsId, m_gridSpacingId, m_gridDimId, m_energyGridId});

        m_tuner->SetLauncher(m_kernel, [this](ktt::ComputeInterface& interface)
        {
            uint64_t sleep = sleepDuration;
            if (randomizeSleep)
                sleep = (sleep*rand())/RAND_MAX;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
            interface.RunKernel(m_definition);
        });
    }

    void InitTuningSpace() override
    {
        m_tuner->AddParameter(m_kernel, "DUMMY_1", vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        m_tuner->AddParameter(m_kernel, "DUMMY_2", vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        m_tuner->AddParameter(m_kernel, "DUMMY_3", vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        m_tuner->AddParameter(m_kernel, "DUMMY_4", vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Dummy> dummy = Dummy::Create<Dummy>(argc, argv, 0, "Examples/Dummy", "Dummy");
    dummy->Run();

    return 0;
}
