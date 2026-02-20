#include <assert.h>
#include <Ktt.h>
#include "Example.h"

std::string getKernelFilePath(std:: string exampleFolderPath, std::string baseName)
{
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

    return kernelPrefix + exampleFolderPath + "/" + baseName + defaultKernelFileSuffix;
}

void Example::Run() 
{
    // Perform tuning
    const auto results = m_tuner.Tune(m_kernel/*, std::make_unique<ktt::ConfigurationCount>(1)*/);
    m_tuner.SaveResults(results, "TranspositionOutput", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "TranspositionOutput", ktt::OutputFormat::JSON);
}

Example::Example(int argc, char** argv, 
                 int defaultProblemSize, 
                 std::string exampleFolderPath,
                 std::string defaultKernelFileBaseName, 
                 std::string defaultReferenceKernelFileBaseName,
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
        argc >= 2 ? std::stoul(std::string(argv[1])) : 0, // Get platform index.
        argc >= 3 ? std::stoul(std::string(argv[2])) : 0, // Get device index.
        m_computeApi
    )
{
    assert(argv != NULL);

    m_problemSize = defaultProblemSize; // In MiB
    if (argc >= 4)
    {
      m_problemSize = atoi(argv[3]);
    }

    m_kernelFile = getKernelFilePath(exampleFolderPath, defaultKernelFileBaseName);
    if (argc >= 5)
    {
        m_kernelFile = std::string(argv[4]);
    }

    m_referenceKernelFile = getKernelFilePath(exampleFolderPath, defaultReferenceKernelFileBaseName);
    if (argc >= 6)
    {
        m_referenceKernelFile = std::string(argv[5]);
    }

    if (m_useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner.SetProfiling(true);
    }
  
    // Create tuner object for chosen platform and device
    m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    InitData();
    InitKernels();
    InitReference();
    InitKernelArguments();
    InitTuningParameters();
}
