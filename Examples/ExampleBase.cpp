#include <assert.h>
#include <Ktt.h>
#include <vector>
#include "ExampleBase.h"

using namespace std;

string ExampleBase::GetKernelFilePath(string exampleFolderPath, string baseName)
{
#if defined(_MSC_VER)
    const string kernelPrefix = "../";
#else
    const string kernelPrefix = "../../";
#endif

#if KTT_CUDA_EXAMPLE
    const string defaultKernelFileSuffix = ".cu";
#elif KTT_OPENCL_EXAMPLE
    const string defaultKernelFileSuffix = ".cl";
#endif

    return kernelPrefix + exampleFolderPath + "/" + baseName + defaultKernelFileSuffix;
}

void ExampleBase::Run() 
{
    // Perform tuning
    const auto results = m_tuner.Tune(m_kernel, GetStopCondition());
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::JSON);
}

ExampleBase::ExampleBase(
    int argc,
    char** argv, 
    int defaultProblemSize, 
    string exampleFolderPath,
    string defaultKernelFileBaseName,
    bool rapidTest,
    bool useProfiling
):
    #if KTT_CUDA_EXAMPLE
    m_computeApi(ktt::ComputeApi::CUDA),
    #elif KTT_OPENCL_EXAMPLE
    m_computeApi(ktt::ComputeApi::OpenCL),
    #endif
    m_rapidTest(rapidTest),
    m_useProfiling(useProfiling),
    m_tuner(
        argc >= 2 ? stoul(string(argv[1])) : 0, // Get platform index.
        argc >= 3 ? stoul(string(argv[2])) : 0, // Get device index.
        m_computeApi
    )
{
    assert(argv != NULL);

    m_problemSize = defaultProblemSize; // In MiB
    if (argc >= 4)
    {
      m_problemSize = atoi(argv[3]);
    }

    m_kernelFile = GetKernelFilePath(exampleFolderPath, defaultKernelFileBaseName);
    if (argc >= 5)
    {
        m_kernelFile = string(argv[4]);
    }

    if (m_useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner.SetProfiling(true);
    }
  
    // Create tuner object for chosen platform and device
    m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

}

void ExampleBase::PostInitialize() 
{
    InitData();
    InitKernels();
    InitTuningParameters();
    InitSearcher();
}

void ExampleBase::InitSearcher() 
{
    // Not necessary, since DS is the default. Demonstrates how a searcher can be set.
    // TODO: Should be empty?
    m_tuner.SetSearcher(m_kernel, std::make_unique<ktt::DeterministicSearcher>());
}

unique_ptr<ktt::StopCondition> ExampleBase::GetStopCondition() 
{
    return nullptr;
}

void ExampleBase::FillBuffers(const vector<vector<float>*> &buffers) 
{
    random_device device;
    default_random_engine engine(device());
    uniform_real_distribution<float> distribution(0.0f, 10.0f);
    for (vector<float> *buffer : buffers) 
    {
        for (size_t i = 0; i < buffer->size(); ++i) 
        {
           buffer->at(i) = distribution(engine); 
        }
    } 
}

void ExampleBase::InitKernelDefault(const string &kernelFunctionName, const string &kernelName,
                                 const ktt::DimensionVector &ndRangeDimensions, const vector<ktt::ArgumentId> &arguments)
{
    // Create m_kernel and configure input/output
    m_definition = m_tuner.AddKernelDefinitionFromFile(kernelFunctionName, m_kernelFile, ndRangeDimensions,
        ktt::DimensionVector(1, 1));
    m_tuner.SetArguments(m_definition, arguments);
        
    m_kernel = m_tuner.CreateSimpleKernel(kernelName, m_definition);
    
}

