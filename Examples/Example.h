#include "ComputeEngine/ComputeApi.h"
#include <Ktt.h>

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

class Example {
public:
    void Run();

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


    Example(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, std::string defaultReferenceKernelFileBaseName = "",
            bool rapidTest = false, bool useProfiling = true);

    virtual void InitKernels();
    virtual void InitReference();
    virtual void InitKernelArguments();
    virtual void InitTuningParameters();
};
