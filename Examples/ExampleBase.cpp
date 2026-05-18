#include "ExampleBase.h"
#include "ComputeEngine/ComputeApi.h"
#include "ExampleConfigurator.h"
#include <assert.h>
#include <vector>
#include <iostream>

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
    const auto results = m_tuner.Tune(m_kernel, std::move(m_config->stopCondition));
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::JSON);

    double bestDuration = numeric_limits<double>::max();
    string bestConfig;
    int successfulRuns = 0;
    for (const auto& result : results)
    {
        double duration = result.GetTotalDuration();
        string config = result.GetConfiguration().GetString();

        cout << "Configuration: " << config
             << " -> Duration: " << duration << " ns"
             << " -> Status: " << (result.GetStatus() == ktt::ResultStatus::Ok ? "OK" : "FAILED")
             << endl;

        if (result.GetStatus() == ktt::ResultStatus::Ok)
        {
            successfulRuns++;
            if (duration < bestDuration) {
                bestDuration = duration;
                bestConfig = config;
            }
        }
    }

    if (!results.empty())
    {
        cout << "\nBest configuration: " << bestConfig << " with duration " << bestDuration << " ns ("
             << successfulRuns << "/" << results.size() << " successful runs)" << endl;
    }
}

ExampleBase::ExampleBase(
    shared_ptr<ExampleConfiguration> config,
    int defaultProblemSize,
    string exampleFolderPath,
    string defaultKernelFileBaseName
) :
    #if KTT_CUDA_EXAMPLE
    m_computeApi(ktt::ComputeApi::CUDA),
    #elif KTT_OPENCL_EXAMPLE
    m_computeApi(ktt::ComputeApi::OpenCL),
    #endif
    m_config(config),
    m_tuner(config->platform, config->device, m_computeApi)
{
    m_problemSize = config->problemSize >= 0 ? config->problemSize : defaultProblemSize;
    m_kernelFile = config->kernelFile.empty()
        ? GetKernelFilePath(exampleFolderPath, defaultKernelFileBaseName)
        : config->kernelFile;
    

    if (config->useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner.SetProfiling(true);
    }

    m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
}

void ExampleBase::PostInitialize() 
{
    InitData();
    InitKernel();
    InitTuningSpace();
    InitSearcher();
}

void ExampleBase::UseFastMath()
{
    if (m_computeApi == ktt::ComputeApi::OpenCL)
    {
        m_tuner.SetCompilerOptions("-cl-fast-relaxed-math");
    }
    else if (m_computeApi == ktt::ComputeApi::CUDA)
    {
        m_tuner.SetCompilerOptions("-use_fast_math");
    }
}

void ExampleBase::UseOpenMP()
{
    if (m_computeApi == ktt::ComputeApi::Cpp)
    {
        m_tuner.SetCompilerOptions("-march=native -fopenmp");
    }
}

void ExampleBase::InitSearcher()
{
    if (!m_config->profileSearchModelPath.empty()) {
        if (m_computeApi != ktt::ComputeApi::CUDA) {
            cerr << "Profile-based search can only be enabled on a CUDA device.\n";
            exit(1);
        }
        m_tuner.SetProfileBasedSearcher(m_kernel, std::move(m_config->profileSearchModelPath));
    } else if (m_config->searcher != nullptr) {
        m_tuner.SetSearcher(m_kernel, std::move(m_config->searcher));
    }
}

void ExampleBase::InitKernelDefault(const string &kernelFunctionName, const string &kernelName,
                                 const ktt::DimensionVector &ndRangeDimensions, const vector<ktt::ArgumentId> &arguments)
{
    // Create m_kernel and configure input/output
    m_definition = m_tuner.AddKernelDefinitionFromFile(kernelFunctionName, m_kernelFile, ndRangeDimensions,
        ktt::DimensionVector());
    m_tuner.SetArguments(m_definition, arguments);
        
    m_kernel = m_tuner.CreateSimpleKernel(kernelName, m_definition);
    
}

