#pragma once

#include "CliComponent.h"
#include "CompilerTuningComponent.h"
#include "RunStats.hpp"
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
    ExampleBase(int argc, char** argv, 
        std::string exampleFolderPath, std::string defaultKernelFileBaseName);

public:
    void Run();

    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, 
            std::string exampleFolderPath, std::string defaultKernelFileBaseName)
    {
        std::unique_ptr<T> ex(new T(argc, argv, exampleFolderPath, defaultKernelFileBaseName));
        ex->PostInitialize();
        return ex;
    }

    virtual ~ExampleBase() = default;

protected:
    const ktt::ComputeApi m_computeApi;
    int m_argc;
    char **m_argv;

    std::string m_kernelFile;
    std::shared_ptr<ktt::Tuner> m_tuner;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    CliComponent m_cli;
    std::unique_ptr<CompilerTuningComponent> m_compilerTuning;

    std::optional<ktt::PreciseMeasurementParameters> m_preciseParams;
    float m_preheatingSeconds;
    std::unique_ptr<ktt::StopCondition> m_stopCondition;
    
    ktt::PlatformIndex m_platform = 0;
    ktt::PlatformIndex m_device = 0;
    bool m_useProfiling = 0;
    bool m_rapidTest = 0;
    bool m_useDynamicTuning = 0;
    double m_dynamicTuningTime = 0;
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
    void Preheat();
    void InitSearcher();

    void RunDynamic();
    void RunOffline();

    bool m_useFastMath = false;
    void UseFastMath();
    bool m_useOpenMP = false;
    void UseOpenMP();

    void UseCompilerTuning();
    void UseInputSizeOption(int numDimensions, ktt::DimensionVector &inputSize);

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
