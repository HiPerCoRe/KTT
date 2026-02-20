#pragma once

#include <Ktt.h>

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
        std::shared_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                                    defaultReferenceKernelFileBaseName, rapidTest, useProfiling));
        ex->PostInitialize();
        return ex;
    }

protected:
    const ktt::ComputeApi m_computeApi;
    // Toggle rapid test (e.g., disable output validation).
    const bool m_rapidTest;
    // Toggle kernel profiling.
    const bool m_useProfiling;

    int m_width;
    int m_height;

    std::string m_kernelFile;
    std::string m_referenceKernelFile;
    ktt::Tuner m_tuner;
    int m_problemSize;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    virtual void PostInitialize();
    virtual void InitData();
    virtual void InitKernels();
    virtual void InitKernelArguments();
    virtual void InitReference();
    virtual void InitTuningParameters();

    void FillBuffers(const std::vector<std::vector<float>*> &buffers);

    void InitReferenceDefault(const std::vector<ktt::ArgumentId> &outputArguments, const ktt::KernelId refKernel);

    struct ReferenceParameters 
    {
        const std::string &functionName;
        const std::string &name;
        ktt::KernelDefinitionId &definition;
        ktt::KernelId &kernel; 
        
        int workGroupWidth;
        int workGroupHeight;
    };
    void InitKernelsDefault(const std::string &kernelFunctionName, const std::string &kernelName,
                            const ReferenceParameters *refParams = nullptr);
};
