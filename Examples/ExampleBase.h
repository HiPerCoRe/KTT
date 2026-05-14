#pragma once

#include "KttTypes.h"
#include <Ktt.h>
#include <random>
#include <type_traits>

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

class ExampleBase {
protected:
    ExampleBase(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, bool rapidTest, bool useProfiling);

public:
    void Run();

    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath,
            std::string defaultKernelFileBaseName, bool rapidTest = false, bool useProfiling = false)
    {
        std::unique_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                                    rapidTest, useProfiling));
        ex->PostInitialize();
        return ex;
    }

protected:
    const ktt::ComputeApi m_computeApi;
    // Toggle rapid test (e.g., disable output validation).
    const bool m_rapidTest;
    // Toggle kernel profiling.
    const bool m_useProfiling;

    std::string m_kernelFile;
    ktt::Tuner m_tuner;
    int m_problemSize;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    static std::string GetKernelFilePath(std::string exampleFolderPath, std::string baseName);

    virtual void PostInitialize();
    virtual void InitData() = 0;
    virtual void InitKernel() = 0;
    virtual void InitTuningSpace() = 0;
    virtual void InitSearcher();

    virtual std::unique_ptr<ktt::StopCondition> GetStopCondition();

    void UseFastMath();
    void UseOpenMP();

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
