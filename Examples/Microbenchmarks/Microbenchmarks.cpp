/******************************************************************************
 * This is a set of microbenchmarks -- they stress different components of
 * a GPU
 * The reason of existence of this example is testing speed and power
 * consumption of used GPU.
 */

#include "../ExampleBase.h"
#include "ComputeEngine/GlobalSizeType.h"
#include "KttTypes.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>

using namespace std;

class Microbenchmarks : public ExampleBase {
protected:
    Microbenchmarks(shared_ptr<ExampleConfiguration> config, int defaultProblemSize,
                    string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        m_kernelSize = 1024 * 1024 * 32;
        m_touchedDataSizeL3 = 1024 * 1024 * 1024;  // Exceed L2
        m_touchedDataSizeL2 = 1024 * 1024;          // Fit L2, exceed L1
        m_touchedDataSizeL1 = 16 * 1024;            // Fit L1

        m_repeatsL3 = (m_touchedDataSizeL3 * 4) / m_kernelSize;
        m_repeatsL2 = (m_touchedDataSizeL2 * 4) / m_kernelSize;
        m_repeatsL1 = (m_touchedDataSizeL1 * 4) / m_kernelSize;

        m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
        UseFastMath();
    }

    friend ExampleBase;

    int m_kernelSize;
    int m_touchedDataSizeL3;
    int m_touchedDataSizeL2;
    int m_touchedDataSizeL1;

    int m_repeatsL3;
    int m_repeatsL2;
    int m_repeatsL1;

    vector<int> m_input;
    vector<int> m_output;

    ktt::KernelId m_kernelL3;
    ktt::KernelId m_kernelL2;
    ktt::KernelId m_kernelL1;

    ktt::KernelDefinitionId m_defL3;
    ktt::KernelDefinitionId m_defL2;
    ktt::KernelDefinitionId m_defL1;
    

    const ktt::KernelResult& getBestResult(const std::vector<ktt::KernelResult>& res) {
        ktt::Nanoseconds bestTime = res[0].GetKernelDuration();
        int bestIdx = 0;
        for (unsigned int i = 0; i < res.size(); i++)
            if (res[i].GetKernelDuration() < bestTime) {
                bestTime = res[i].GetKernelDuration();
                bestIdx = i;
            }
        return res[bestIdx];
    }

    void reportKernelStats(ktt::KernelResult& bestTime, long long int sumTransferred){
        #if KTT_POWER_USAGE_NVML
            double energy = bestTime.GetResults()[0].GetEnergyConsumption();
            uint64_t watts = bestTime.GetResults()[0].GetPowerUsage()/1000.0;
        #else
            double energy = -1;
            uint64_t watts = -1;
        #endif
        std::cout << "Configuration: "
            << bestTime.GetConfiguration().GetString() << "\n"
            << "Time, energy, and power: "
            << bestTime.GetKernelDuration() << " ns, "
            << energy << " J, "
            << watts << " W\n"
            << "Performance: " << sumTransferred / bestTime.GetKernelDuration()
            << "GB/s\n"
            << "Cost of transferring one byte: "
            << 1000000000000.0*energy/sumTransferred << " pJ\n";
    }

    void InitData() override
    {
        const int maxDataSize = max({m_touchedDataSizeL3/4, m_touchedDataSizeL2/4, m_touchedDataSizeL1/4});

        m_input.resize(maxDataSize);
        m_output.resize(max(m_kernelSize, maxDataSize));

        FillBuffers({&m_input}, 0, 100);
    }

    void InitKernel() override
    {
    }

    void InitTuningSpace() override
    {
    }

public:
    void Run()
    {
        m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
        m_tuner.SetLoggingLevel(ktt::LoggingLevel::Warning);
        const long long int copyDataSize = (long long int)(1024*1024*1024)*100; 
        const int repeats = copyDataSize/m_kernelSize;

        const long long int sumTransferred = (long long int)(m_kernelSize) * (long long int)(repeats+1) * 4; //XXX +1 for writes

        const ktt::DimensionVector ndRangeDimensions(m_kernelSize);
        const ktt::DimensionVector workGroupDimensions(1);

        const ktt::KernelDefinitionId defMemstress = m_tuner.AddKernelDefinitionFromFile("stressMem", m_kernelFile, ndRangeDimensions, workGroupDimensions);

        const ktt::KernelId kernelMemstress = m_tuner.CreateSimpleKernel("StressMem", defMemstress);

        const ktt::ArgumentId inputId = m_tuner.AddArgumentVector(m_input, 
            ktt::ArgumentAccessType::ReadOnly);
        const ktt::ArgumentId outputId = m_tuner.AddArgumentVector(m_output, 
            ktt::ArgumentAccessType::WriteOnly);
        const ktt::ArgumentId sizeId = m_tuner.AddArgumentScalar(m_touchedDataSizeL3/4);
        const ktt::ArgumentId repeatsId = m_tuner.AddArgumentScalar(repeats);

        m_tuner.AddParameter(kernelMemstress, "BLOCK", 
            std::vector<uint64_t>{64, 128, 256, 512, 1024});
        m_tuner.AddThreadModifier(kernelMemstress, {defMemstress},
            ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
            ktt::ModifierAction::Multiply);

        m_tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeId, repeatsId});

        std::cout << "\nExecuting memory stress test: transferring "
            << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
            << m_touchedDataSizeL3/1024/1024/1024 << "GB (to exceed L2)...";
        fflush(stdout);
        const auto results = m_tuner.Tune(kernelMemstress/*, std::make_unique<ktt::ConfigurationCount>(1)*/);
        m_tuner.SaveResults(results, "MemStressOutputGlobal", ktt::OutputFormat::JSON);
        m_tuner.SaveResults(results, "MemStressOutputGlobal", ktt::OutputFormat::XML);

        ktt::KernelResult bestTime = getBestResult(results);
        std::cout << "done.\n";
        reportKernelStats(bestTime, sumTransferred);

        std::cout << "\nExecuting memory stress test: transferring "
            << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
            << m_touchedDataSizeL2/1024/1024 << "MB (to fit L2 but exceed L1)...";
        fflush(stdout);
        const ktt::KernelId kernelMemstressL2 = m_tuner.CreateSimpleKernel("StressMemL2", defMemstress);
        m_tuner.AddParameter(kernelMemstressL2, "BLOCK",
            std::vector<uint64_t>{64, 128, 256, 512, 1024});
        m_tuner.AddThreadModifier(kernelMemstressL2, {defMemstress},
            ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
            ktt::ModifierAction::Multiply);
        const ktt::ArgumentId sizeL2Id = m_tuner.AddArgumentScalar(m_touchedDataSizeL2/4);
        m_tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeL2Id, repeatsId});

        const auto resultsL2 = m_tuner.Tune(kernelMemstressL2);
        m_tuner.SaveResults(results, "MemStressOutputL2", ktt::OutputFormat::JSON);
        m_tuner.SaveResults(results, "MemStressOutputL2", ktt::OutputFormat::XML);
        ktt::KernelResult bestTimeL2 = getBestResult(resultsL2);
        std::cout << "done.\n";
        reportKernelStats(bestTimeL2, sumTransferred);

        std::cout << "\nExecuting memory stress test: transferring "
            << sumTransferred/1024/1024/1024 << "GB of memory at footprint "
            << m_touchedDataSizeL1/1024 << "KB (to fit L1)...";
        fflush(stdout);
        const ktt::KernelId kernelMemstressL1 = m_tuner.CreateSimpleKernel("StressMemL1", defMemstress);
        m_tuner.AddParameter(kernelMemstressL1, "BLOCK",
            std::vector<uint64_t>{64, 128, 256, 512, 1024});
        m_tuner.AddThreadModifier(kernelMemstressL1, {defMemstress},
            ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
            ktt::ModifierAction::Multiply);
        const ktt::ArgumentId sizeL1Id = m_tuner.AddArgumentScalar(m_touchedDataSizeL1/4);
        m_tuner.SetArguments(defMemstress, std::vector<ktt::ArgumentId>{inputId, outputId, sizeL1Id, repeatsId});

        const auto resultsL1 = m_tuner.Tune(kernelMemstressL1);
        m_tuner.SaveResults(results, "MemStressOutputL1", ktt::OutputFormat::JSON);
        m_tuner.SaveResults(results, "MemStressOutputL1", ktt::OutputFormat::XML);
        ktt::KernelResult bestTimeL1 = getBestResult(resultsL1);
        std::cout << "done.\n";
        reportKernelStats(bestTimeL1, sumTransferred);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Microbenchmarks> microbench = Microbenchmarks::Create<Microbenchmarks>(
        argc, argv, 0, "Examples/Microbenchmarks", "Microbenchmarks");
    microbench->Run();

    return 0;
}
