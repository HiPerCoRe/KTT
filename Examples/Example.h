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
    ktt::Tuner m_tuner;
    int m_problemSize;
    int m_bufferSize;


    Example(int argc, char** argv, std::string exampleFolderPath, int defaultProblemSize, 
            std::string defaultKernelFileBaseName, std::string defaultRererenceKernelFileBaseName = "",
            bool rapidTest = false, bool useProfiling = true);

    virtual void InitKernels();
    virtual void InitReference();
    virtual void InitKernelArguments();
    virtual void InitTuningParameters();
};
