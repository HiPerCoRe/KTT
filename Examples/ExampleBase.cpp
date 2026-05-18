#include "ExampleBase.h"
#include "Api/Output/KernelResult.h"
#include "ComputeEngine/ComputeApi.h"
#include "ExampleConfigurator.h"
#include "Utility/Logger/Logger.h"
#include "Utility/Logger/LoggingLevel.h"
#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

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

void ExampleBase::PrintRunStats(const string& phaseName, const RunStats& stats, double throughput)
{
    cout << "\n--- " << phaseName << " complete ---" << endl;
    cout << "Total runs: " << stats.totalRuns << endl;
    cout << "Successful runs: " << stats.successfulRuns << "/" << stats.totalRuns << endl;
    if (!stats.bestConfig.empty()) {
        cout << "Best configuration: " << stats.bestConfig << endl;
        cout << "Best duration: " << stats.bestDuration << " ns" << endl;
    }
    cout << "Throughput: " << fixed << setprecision(2) << throughput << " runs/s" << endl;
    cout.unsetf(ios_base::floatfield);
    cout.precision(6);
}

void ExampleBase::PrintProgress(const string& phaseName, int currentRun, double elapsedSeconds,
                                double timeBudget, double bestDuration, double throughput)
{
    cout << "[" << phaseName << "] Run " << currentRun
         << " | Elapsed: " << elapsedSeconds << "s / " << timeBudget << "s"
         << " | Best: " << bestDuration << " ns"
         << " | " << fixed << setprecision(2) << throughput << " runs/s" << endl;
    cout.unsetf(ios_base::floatfield);
    cout.precision(6);
}

void ExampleBase::RunStats::Update(ktt::KernelResult result) {
    totalRuns++;

    if (result.GetStatus() == ktt::ResultStatus::Ok) {
        successfulRuns++;
        double duration = result.GetTotalDuration();
        if (duration < bestDuration) {
            bestDuration = duration;
            bestConfig = result.GetConfiguration().GetString();
        }
    }
}

ExampleBase::RunStats ExampleBase::RunTuningPhase(
    const std::chrono::steady_clock::time_point& startTime, double timeBudgetSeconds, int printInterval)
{
    RunStats stats;

    while (true) {
        const auto currentTime = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();

        if (elapsedSeconds >= timeBudgetSeconds) {
            break;
        }

        const auto result = m_tuner.TuneIteration(m_kernel, {}, false, m_config->preciseParams);
        stats.Update(result);

        if (stats.totalRuns % printInterval == 0 || stats.totalRuns == 1) {
            double throughput = elapsedSeconds > 0 ? stats.totalRuns / elapsedSeconds : 0;
            PrintProgress("Tuning", stats.totalRuns, elapsedSeconds, timeBudgetSeconds,
                          stats.bestDuration, throughput);
        }
    }

    return stats;
}

ExampleBase::RunStats ExampleBase::RunExecutionPhase(
    const std::chrono::steady_clock::time_point& startTime, double timeBudgetSeconds,
    const ktt::KernelConfiguration& bestConfig, int printInterval)
{
    RunStats stats;

    while (true) {
        const auto currentTime = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();

        if (elapsedSeconds >= timeBudgetSeconds) {
            break;
        }

        const auto result = m_tuner.Run(m_kernel, bestConfig, {});
        stats.Update(result);

        if (stats.totalRuns % printInterval == 0) {
            double throughput = elapsedSeconds > 0 ? stats.totalRuns / elapsedSeconds : 0;
            PrintProgress("Run", stats.totalRuns, elapsedSeconds, timeBudgetSeconds,
                          stats.bestDuration, throughput);
        }
    }

    return stats;
}

void ExampleBase::RunDynamic()
{
    ktt::Logger::GetLogger().SetLoggingLevel(ktt::LoggingLevel::Warning);
    const auto startTime = std::chrono::steady_clock::now();
    const double timeBudgetSeconds = m_config->dynamicTuningTime > 0 ? m_config->dynamicTuningTime : 60.0;
    constexpr int printInterval = 50;

    RunStats tuningStats = RunTuningPhase(startTime, timeBudgetSeconds, printInterval);

    const auto tuningPhaseEnd = std::chrono::steady_clock::now();
    double tuningElapsed = std::chrono::duration<double>(tuningPhaseEnd - startTime).count();
    double tuningThroughput = tuningElapsed > 0 ? tuningStats.totalRuns / tuningElapsed : 0;

    PrintRunStats("Tuning phase", tuningStats, tuningThroughput);

    const auto bestConfigData = m_tuner.GetBestConfiguration(m_kernel);

    cout << "\n--- Running with best configuration ---" << endl;
    const auto runStartTime = std::chrono::steady_clock::now();
    RunStats runStats = RunExecutionPhase(runStartTime, timeBudgetSeconds, bestConfigData, printInterval);

    const auto totalEndTime = std::chrono::steady_clock::now();
    double runElapsed = std::chrono::duration<double>(totalEndTime - runStartTime).count();
    double runThroughput = runElapsed > 0 ? runStats.totalRuns / runElapsed : 0;

    PrintRunStats("Final statistics", runStats, runThroughput);
}

void ExampleBase::RunOffline()
{
    const auto startTime = std::chrono::steady_clock::now();

    const auto results = m_tuner.Tune(m_kernel, std::move(m_config->stopCondition), m_config->preciseParams);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "Output", ktt::OutputFormat::JSON);

    const auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    double throughput = elapsed > 0 ? static_cast<double>(results.size()) / elapsed : 0;

    RunStats stats;
    for (const auto& result : results)
    {
        if (result.GetStatus() == ktt::ResultStatus::Ok)
        {
            stats.successfulRuns++;
            if (result.GetTotalDuration() < stats.bestDuration) {
                stats.bestDuration = result.GetTotalDuration();
                stats.bestConfig = result.GetConfiguration().GetString();
            }
        }
    }
    stats.totalRuns = static_cast<int>(results.size());

    PrintRunStats("Offline tuning", stats, throughput);
}

void ExampleBase::Run()
{
    if (m_config->useDynamicTuning) RunDynamic();
    else RunOffline();
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

