#include <assert.h>
#include <Ktt.h>
#include <vector>
#include "Example.h"

using namespace std;

string getKernelFilePath( string exampleFolderPath, string baseName)
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

void Example::Run() 
{
    // Perform tuning
    const auto results = m_tuner.Tune(m_kernel/*, make_unique<ktt::ConfigurationCount>(1)*/);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::JSON);
}

Example::Example(int argc, char** argv, 
                 int defaultProblemSize, 
                 string exampleFolderPath,
                 string defaultKernelFileBaseName, 
                 string defaultReferenceKernelFileBaseName,
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
    m_width = m_problemSize * 1024 * 1024;
    m_height = 1;

    m_kernelFile = getKernelFilePath(exampleFolderPath, defaultKernelFileBaseName);
    if (argc >= 5)
    {
        m_kernelFile = string(argv[4]);
    }

    m_referenceKernelFile = getKernelFilePath(exampleFolderPath, defaultReferenceKernelFileBaseName);
    if (argc >= 6)
    {
        m_referenceKernelFile = string(argv[5]);
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

void Example::PostInitialize() 
{
    InitData();
    InitKernels();
    InitKernelArguments();
    InitReference();
    InitTuningParameters();
}

void Example::InitData()
{
    assert(false && "Abstract method from Example must be implemented");
}

void Example::InitKernels() 
{
    assert(false && "Abstract method from Example must be implemented");
}

void Example::InitReference() 
{
    assert(false && "Abstract method from Example must be implemented");
}

void Example::InitKernelArguments() 
{
    assert(false && "Abstract method from Example must be implemented");
}

void Example::InitTuningParameters() 
{
    assert(false && "Abstract method from Example must be implemented");
}

void Example::FillBuffers(const vector<vector<float>*> &buffers) 
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

void Example::InitKernelsDefault(const string &kernelFunctionName, const string &kernelName,
                                 const ReferenceParameters *refParams)
{
    const ktt::DimensionVector ndRangeDimensions(m_width, m_height);

    // Create m_kernel and configure input/output
    m_definition = m_tuner.AddKernelDefinitionFromFile(kernelFunctionName, m_kernelFile, ndRangeDimensions,
        ktt::DimensionVector(1, 1));
    m_kernel = m_tuner.CreateSimpleKernel(kernelName, m_definition);
    
    if (refParams == nullptr) return;
    const ktt::DimensionVector ndRangeDimensionsReference(m_width / refParams->workGroupWidth, m_height / refParams ->workGroupHeight);
    const ktt::DimensionVector referenceWorkGroupDimensions(refParams->workGroupWidth, refParams->workGroupHeight);

    refParams->definition = m_tuner.AddKernelDefinitionFromFile(refParams->functionName, m_referenceKernelFile,
        ndRangeDimensionsReference, referenceWorkGroupDimensions);
    refParams->kernel = m_tuner.CreateSimpleKernel(refParams->name, refParams->definition);
}

void Example::InitReferenceDefault(const vector<ktt::ArgumentId> &outputArguments, ktt::KernelId refKernel) 
{
    if (!m_rapidTest)
    {
        for (auto arg : outputArguments) {
            m_tuner.SetReferenceKernel(arg, refKernel, ktt::KernelConfiguration());
        }
        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
    }
}
