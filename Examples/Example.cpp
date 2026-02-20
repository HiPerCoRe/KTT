#include <assert.h>
#include <Ktt.h>
#include "Example.h"

Example::Example(int argc, char** argv, 
                 std::string exampleFolderPath, 
                 int defaultProblemSize,
                 std::string defaultKernelFileBaseName, 
                 std::string defaultRererenceKernelFileBaseName,
                 bool rapidTest,
                 bool useProfiling):
    #if KTT_CUDA_EXAMPLE
    m_computeApi(ktt::ComputeApi::CUDA),
    #elif KTT_OPENCL_EXAMPLE
    m_computeApi(ktt::ComputeApi::OpenCL),
    #endif
    m_rapidTest(rapidTest),
    m_useProfiling(useProfiling),
    m_tuner(
        argc >= 2 ? std::stoul(std::string(argv[1])) : 0,
        argc >= 3 ? std::stoul(std::string(argv[2])) : 0,
        m_computeApi
    )
{
    assert(argv != NULL);

#if defined(_MSC_VER)
    const std::string kernelPrefix = "";
#else
    const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFileSuffix = ".cu";
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFileSuffix = ".cl";
#endif

    m_kernelFile = kernelPrefix + exampleFolderPath + "/" + defaultKernelFileBaseName + defaultKernelFileSuffix;
    if (argc >= 4)
    {
        m_kernelFile = std::string(argv[3]);
    }

    m_problemSize = defaultProblemSize; // In MiB
    if (argc >= 5)
    {
      m_problemSize = atoi(argv[4]);
    }

    if (m_useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner.SetProfiling(true);
    }
  
    // Create tuner object for chosen platform and device
    m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
}
