#pragma once

#include <Ktt.h>
#include <memory>

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

class Example {
protected:
    Example(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, std::string defaultReferenceKernelFileBaseName,
            bool rapidTest, bool useProfiling);

public:
    void Run();

    template <class T>
    static std::shared_ptr<T> Create(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, std::string defaultReferenceKernelFileBaseName = "",
            bool rapidTest = false, bool useProfiling = false)
    {
        auto ex = std::make_unique<T>(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                                 defaultReferenceKernelFileBaseName, rapidTest, useProfiling);
        ex->InitData();
        ex->InitKernels();
        ex->InitReference();
        ex->InitKernelArguments();
        ex->InitTuningParameters();
        return ex;
    }

protected:
    const ktt::ComputeApi m_computeApi;
    // Toggle rapid test (e.g., disable output validation).
    const bool m_rapidTest;
    // Toggle kernel profiling.
    const bool m_useProfiling;

    std::string m_kernelFile;
    std::string m_referenceKernelFile;
    ktt::Tuner m_tuner;
    int m_problemSize;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    virtual void InitData();
    virtual void InitKernels();
    virtual void InitReference();
    virtual void InitKernelArguments();
    virtual void InitTuningParameters();

    void InitReferenceDefault(std::vector<ktt::ArgumentId> outputArguments, ktt::KernelId refKernel);
};
