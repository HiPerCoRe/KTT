#pragma once

#include "Api/Configuration/PreciseMeasurementParameters.h"
#include "Api/Searcher/Searcher.h"
#include "Api/StopCondition/StopCondition.h"
#include "CliComponent.h"
#include "RunStats.hpp"
#include "KttTypes.h"
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
    ExampleBase(int argc, char** argv, int defaultProblemSize, 
        std::string exampleFolderPath, std::string defaultKernelFileBaseName);

public:
    void Run();

    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, int defaultProblemSize, 
            std::string exampleFolderPath, std::string defaultKernelFileBaseName)
    {
        std::unique_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName));
        ex->PostInitialize();
        return ex;
    }

protected:
    const ktt::ComputeApi m_computeApi;
    int m_argc;
    char **m_argv;

    std::string m_kernelFile;
    std::unique_ptr<ktt::Tuner> m_tuner;
    int m_problemSize;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    CliComponent m_cli;

    std::optional<ktt::PreciseMeasurementParameters> m_preciseParams;
    std::unique_ptr<ktt::StopCondition> m_stopCondition;
    
    ktt::PlatformIndex m_platform;
    ktt::PlatformIndex m_device;
    bool m_useProfiling;
    bool m_rapidTest;
    bool m_useDynamicTuning;
    double m_dynamicTuningTime;
    std::unique_ptr<ktt::Searcher> m_searcher;
    std::string m_profileSearchModelPath;

    static std::string GetKernelFilePath(std::string exampleFolderPath, std::string baseName, std::optional<std::string> suffix = std::nullopt);

    virtual void PostInitialize();
    virtual void InitCLI();
    void ProcessCLI();
    void InitTuner();
    virtual void InitData() = 0;
    virtual void InitKernel() = 0;
    virtual void InitTuningSpace() = 0;
    void InitSearcher();

    void RunDynamic();
    void RunOffline();

    void UseFastMath();
    void UseOpenMP();

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
