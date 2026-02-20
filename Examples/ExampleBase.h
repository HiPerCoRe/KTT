#pragma once

#include "KttTypes.h"
#include <Ktt.h>

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
    static std::shared_ptr<T> Create(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, bool rapidTest = false, bool useProfiling = false)
    {
        std::shared_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
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
    virtual void InitKernels() = 0;
    virtual void InitTuningParameters() = 0;
    virtual void InitSearcher();

    virtual std::unique_ptr<ktt::StopCondition> GetStopCondition();

    void FillBuffers(const std::vector<std::vector<float>*> &buffers);


    void InitKernelDefault(const std::string &kernelFunctionName, const std::string &kernelName,
                            const ktt::DimensionVector &ndRangeDimensions, const std::vector<ktt::ArgumentId> &arguments);
};
