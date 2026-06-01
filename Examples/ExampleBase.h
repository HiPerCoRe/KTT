#pragma once

#include "Api/Output/KernelResult.h"
#include "ExampleConfigurator.h"
#include <Ktt.h>
#include <memory>
#include <random>
#include <type_traits>
#include <optional>

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

class ExampleBase {
protected:
    ExampleBase(std::shared_ptr<ExampleConfiguration> config, int defaultProblemSize, 
        std::string exampleFolderPath, std::string defaultKernelFileBaseName);

public:
    void Run();

    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, int defaultProblemSize, 
            std::string exampleFolderPath, std::string defaultKernelFileBaseName)
    {
        auto config = std::make_shared<ExampleConfiguration>(ProcessInput(argc, argv));
        std::unique_ptr<T> ex(new T(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName));
        ex->PostInitialize();
        return ex;
    }

protected:
    const ktt::ComputeApi m_computeApi;
    std::shared_ptr<ExampleConfiguration> m_config;

    std::string m_kernelFile;
    ktt::Tuner m_tuner;
    int m_problemSize;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    static std::string GetKernelFilePath(std::string exampleFolderPath, std::string baseName, std::optional<std::string> suffix = std::nullopt);

    virtual void PostInitialize();
    virtual void InitData() = 0;
    virtual void InitKernel() = 0;
    virtual void InitTuningSpace() = 0;
    void InitSearcher();

    void RunDynamic();
    void RunOffline();

    void UseFastMath();
    void UseOpenMP();

    struct RunStats {
        int totalRuns = 0;
        int successfulRuns = 0;
        double bestDuration = std::numeric_limits<double>::max();
        std::string bestConfig;

        void Update(ktt::KernelResult);
    };

    void PrintRunStats(const std::string& phaseName, const RunStats& stats, double throughput = -1);
    void PrintProgress(const std::string& phaseName, int currentRun, double elapsedSeconds,
                       double timeBudget, double bestDuration, double throughput);

    RunStats RunTuningPhase(const std::chrono::steady_clock::time_point& startTime,
                            double timeBudgetSeconds, int printInterval);
    RunStats RunExecutionPhase(const std::chrono::steady_clock::time_point& startTime,
                               double timeBudgetSeconds, const ktt::KernelConfiguration& bestConfig,
                               int printInterval);

    template <typename T>
    void FillBuffers(const std::vector<std::vector<T>*> &buffers, T minimum = 0, T maximum = 10) 
    {
        std::random_device device;
        std::default_random_engine engine(device());

        static_assert(std::is_arithmetic_v<T>,
                  "FillBuffers accepts only numeric types");
        using Dist = std::conditional_t<
            std::is_floating_point_v<T>,
            std::uniform_real_distribution<T>,
            std::uniform_int_distribution<T>
        >;
        Dist distribution(minimum, maximum);

        for (std::vector<T> *buffer : buffers) 
        {
            for (size_t i = 0; i < buffer->size(); ++i) 
            {
            buffer->at(i) = distribution(engine); 
            }
        } 
    }

    void InitKernelDefault(const std::string &kernelFunctionName, const std::string &kernelName,
                            const ktt::DimensionVector &ndRangeDimensions, const std::vector<ktt::ArgumentId> &arguments);
};
