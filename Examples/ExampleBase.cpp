#include "ExampleBase.h"
#include "Api/Output/KernelResult.h"
#include "ComputeEngine/ComputeApi.h"
#include "CliComponent.h"
#include "Utility/Logger/Logger.h"
#include "Utility/Logger/LoggingLevel.h"
#include <assert.h>
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

string ExampleBase::GetKernelFilePath(string exampleFolderPath, string baseName, optional<string> suffix)
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
#elif KTT_CPP_EXAMPLE
    const string defaultKernelFileSuffix = ".cppkernel";
#endif

    return kernelPrefix + exampleFolderPath + "/" + baseName + (suffix == nullopt ?defaultKernelFileSuffix : suffix.value());
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

RunStats ExampleBase::RunTuningPhase(
    const std::chrono::steady_clock::time_point& startTime, double timeBudgetSeconds, int printInterval)
{
    RunStats stats;

    while (true) {
        const auto currentTime = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();

        if (elapsedSeconds >= timeBudgetSeconds) {
            break;
        }

        const auto result = m_tuner->TuneIteration(m_kernel, {}, false, m_preciseParams);
        stats.Update(result);

        if (stats.totalRuns % printInterval == 0 || stats.totalRuns == 1) {
            double throughput = elapsedSeconds > 0 ? stats.totalRuns / elapsedSeconds : 0;
            PrintProgress("Tuning", stats.totalRuns, elapsedSeconds, timeBudgetSeconds,
                          stats.bestDuration, throughput);
        }
    }

    return stats;
}

RunStats ExampleBase::RunExecutionPhase(
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

        const auto result = m_tuner->Run(m_kernel, bestConfig, {});
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
    const double timeBudgetSeconds = m_dynamicTuningTime > 0 ? m_dynamicTuningTime : 60.0;
    constexpr int printInterval = 50;

    RunStats tuningStats = RunTuningPhase(startTime, timeBudgetSeconds, printInterval);

    const auto tuningPhaseEnd = std::chrono::steady_clock::now();
    double tuningElapsed = std::chrono::duration<double>(tuningPhaseEnd - startTime).count();
    double tuningThroughput = tuningElapsed > 0 ? tuningStats.totalRuns / tuningElapsed : 0;

    tuningStats.Print("Tuning phase", tuningThroughput);

    m_compilerTuning->Run();

    const auto bestConfigData = m_tuner->GetBestConfiguration(m_kernel);

    cout << "\n--- Running with best configuration ---" << endl;
    const auto runStartTime = std::chrono::steady_clock::now();
    RunStats runStats = RunExecutionPhase(runStartTime, timeBudgetSeconds, bestConfigData, printInterval);

    const auto totalEndTime = std::chrono::steady_clock::now();
    double runElapsed = std::chrono::duration<double>(totalEndTime - runStartTime).count();
    double runThroughput = runElapsed > 0 ? runStats.totalRuns / runElapsed : 0;

    runStats.Print("Final statistics", runThroughput);
}

void ExampleBase::RunOffline()
{
    const auto startTime = std::chrono::steady_clock::now();

    const auto results = m_tuner->Tune(m_kernel, std::move(m_stopCondition), m_preciseParams);
    m_tuner->SaveResults(results, "Output", ktt::OutputFormat::XML);
    m_tuner->SaveResults(results, "Output", ktt::OutputFormat::JSON);

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

    stats.Print("Offline tuning", throughput);

    m_compilerTuning->Run();
}

void ExampleBase::Run()
{
    if (m_useDynamicTuning) RunDynamic();
    else RunOffline();
}

ExampleBase::ExampleBase(
    int argc,
    char **argv,
   
    string exampleFolderPath,
    string defaultKernelFileBaseName
) :
    #if KTT_CUDA_EXAMPLE
    m_computeApi(ktt::ComputeApi::CUDA),
    #elif KTT_OPENCL_EXAMPLE
    m_computeApi(ktt::ComputeApi::OpenCL),
    #elif KTT_CPP_EXAMPLE
    m_computeApi(ktt::ComputeApi::Cpp),
    #endif
    m_argc(argc),
    m_argv(argv),
    m_compilerTuning(make_unique<NoCompilerTuning>(m_tuner, m_kernel)),
    m_preheatingSeconds(0)
{
    m_kernelFile = GetKernelFilePath(exampleFolderPath, defaultKernelFileBaseName);
}

void ExampleBase::PostInitialize() 
{
    InitCLI();
    ProcessCLI();
    InitTuner();
    InitData();
    InitKernel();
    InitTuningSpace();
    Preheat();
    InitSearcher();
}

void ExampleBase::InitCLI() {
    m_cli.AddOption({[this](const vector<string> &) {
        m_rapidTest = true;
    }, "--rapidTest", "Run in rapid test mode"});

    m_cli.AddOption({[this](const vector<string> &) {
        m_useProfiling = true;
    }, "--profile", "Enable profiling"});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_platform = stoul(args[0]);
    }, "--platform", "Platform index (expects int)", "<index>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_device = stoul(args[0]);
    }, "--device", "Device index (expects int)", "<index>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_kernelFile = args[0];
    }, "--kernelPath", "Kernel file path (expects string)", "<path>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        if (args[0] == "ds") {
            m_searcher = make_unique<ktt::DeterministicSearcher>();
        } else if (args[0] == "random") {
            m_searcher = make_unique<ktt::RandomSearcher>();
        } else if (args[0] == "mcmc") {
            m_searcher = make_unique<ktt::McmcSearcher>();
        } else {
            cerr << "--searcher expects one of (ds, random, mcmc)\n";
            exit(1);
        }
    }, "--searcher", "Searcher type (ds, random, mcmc)", "<type>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_profileSearchModelPath = args[0];
        m_useProfiling = true;
    }, "--profileSearcher", 
    "Enable profile searcher and set path to model (expects string) (functions only on CUDA devices)",
    "<pathToModel>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        if (args[0] == "confs") {
            m_stopCondition = make_unique<ktt::ConfigurationCount>(stoul(args[1]));
        } else if (args[0] == "fails") {
            m_stopCondition = make_unique<ktt::FailureCount>(stoul(args[1]));
        } else if (args[0] == "time") {
            m_stopCondition = make_unique<ktt::TuningDuration>(stod(args[1]));
        } else if (args[0] == "best") {
            m_stopCondition = make_unique<ktt::ConfigurationDuration>(stod(args[1]));
        } else {
            cerr << "--stopCondition expects one of (confs, fails, time, best)\n";
            exit(1);
        }
    }, "--stopCondition", 
    "Set a stop condition. <type> can be confs, fails, time, best. "
    "<limit> is respectively configuration count (ulong), failed kernel run count (ulong), "
    "total tuning duration in seconds (double), best configuration duration in milliseconds (double).",
    "<type> <limit>", 2});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_preciseParams = ktt::PreciseMeasurementParameters(stoul(args[0]),
            stoul(args[1]), stod(args[2]));
    }, "--preciseParams", "Set PreciseMeasurementParameters, calculationDurationMethod is the default Minimum, refer to KTT documentation for details.",
    "<minTimeMs> <maxTimeMs> <maxPowerDiff>", 3});
    m_cli.AddOption({[this](const vector<string> &args) {
        if (m_preciseParams == std::nullopt) {
            cerr << "--preciseParams must be used before this option.\n";
            exit(1);
        }
        ktt::DurationCalculationMethod calcMethod = ktt::DurationCalculationMethod::Minimum;
        if (args[0] == "min") {}
        else if (args[0] == "median") {
            calcMethod = ktt::DurationCalculationMethod::Median;
        } else if (args[0] == "avg") {
            calcMethod = ktt::DurationCalculationMethod::Average;
        } else {
            cerr << "--preciseParamsCalcMethod expects one of (min, median, avg)\n";
            exit(1);
        }
        m_preciseParams->durationCalculationMethod = calcMethod;
    }, "--preciseParamsCalcMethod", "Optionally set PreciseMeasurementParameters::durationCalculationMethod AFTER USING --preciseParams, expects one of "
    "(min, median, avg), refer to KTT documentation for details.",
    "<calcMethod>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_preheatingSeconds = stof(args[0]);
    }, "--preheat", "Set time in seconds to spend preheating GPU before tuning starts. 0 (disabled) is default. Expects float.", "<time>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        m_useDynamicTuning = true;
        m_dynamicTuningTime = stod(args[0]);
    }, "--useDynamicTuning", "Enables a basic implementation of dynamic tuning."
    "The tuning will last <time> (double) seconds and then the only the best configuration will be run.",
    "<time>", 1});

    m_cli.AddOption({[this](const vector<string> &args) {
        if (args[0] == "off") m_tuner->SetLoggingLevel(ktt::LoggingLevel::Off);
        else if (args[0] == "error") m_tuner->SetLoggingLevel(ktt::LoggingLevel::Error);
        else if (args[0] == "warning") m_tuner->SetLoggingLevel(ktt::LoggingLevel::Warning);
        else if (args[0] == "info") m_tuner->SetLoggingLevel(ktt::LoggingLevel::Info);
        else if (args[0] == "debug") m_tuner->SetLoggingLevel(ktt::LoggingLevel::Debug);
        else {
            cerr << "--loggingLevel expects one of (off, error, warning, info, debug)";
            exit(1);
        }
    }, "--loggingLevel", "Set the logging level, can be one of (off, error, warning, info, debug)", "<level>", 1});

    m_compilerTuning->InitCLIOptions(m_cli);
}

void ExampleBase::ProcessCLI() {
    m_cli.ProcessInput(m_argc, m_argv);
}

void ExampleBase::InitTuner() {
    m_tuner = make_unique<ktt::Tuner>(m_platform, m_device, m_computeApi);
    if (m_useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner->SetProfiling(true);
    }
    m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    m_tuner->SetTimeUnit(ktt::TimeUnit::Microseconds);
}

void ExampleBase::UseFastMath()
{
    if (m_computeApi == ktt::ComputeApi::OpenCL)
    {
        m_tuner->SetCompilerOptions("-cl-fast-relaxed-math");
    }
    else if (m_computeApi == ktt::ComputeApi::CUDA)
    {
        m_tuner->SetCompilerOptions("-use_fast_math");
    }
}

void ExampleBase::UseOpenMP()
{
    if (m_computeApi == ktt::ComputeApi::Cpp)
    {
        m_tuner->SetCompilerOptions("-march=native -fopenmp");
    }
}

void ExampleBase::UseCompilerTuning()
{
    m_compilerTuning = make_unique<CompilerTuningComponent>(m_tuner, m_kernel);
}

void ExampleBase::UseInputSizeOption(int numDimensions, ktt::DimensionVector &inputSize)
{
    assert(numDimensions > 0 && numDimensions <= 3);
    m_cli.AddOption(CliOption({[&inputSize, numDimensions](const vector<string> &args){
        ktt::ModifierDimension dims[] = {ktt::ModifierDimension::X, ktt::ModifierDimension::Y, ktt::ModifierDimension::Z};
        for (int i = 0; i < numDimensions; ++i) {
            inputSize.SetSize(dims[i], stoul(args[i]));
        }
    }, "--inputSize", "Set input size, expects " + to_string(numDimensions) + " int" + (numDimensions > 1 ? "s." : "."),
    string("<sizeX>") + (numDimensions >= 2 ? " <sizeY>" : "") + (numDimensions >= 3 ? " <sizeZ>" : ""), numDimensions}));
}

void ExampleBase::Preheat() {
    if (m_preheatingSeconds > 0) {
        cout << "\n--- Preheating GPU... ---" << endl;
        m_tuner->SetSearcher(m_kernel, make_unique<ktt::RandomSearcher>());
        m_tuner->Tune(m_kernel, make_unique<ktt::TuningDuration>(m_preheatingSeconds));
        m_tuner->ClearConfigurationData(m_kernel);
        cout << "--- Preheating complete ---\n" << endl;
    }
}

void ExampleBase::InitSearcher()
{
    if (!m_profileSearchModelPath.empty()) {
        if (m_computeApi != ktt::ComputeApi::CUDA) {
            cerr << "Profile-based search can only be enabled on a CUDA device.\n";
            exit(1);
        }
        m_tuner->SetProfileBasedSearcher(m_kernel, std::move(m_profileSearchModelPath));
    } else if (m_searcher != nullptr) {
        m_tuner->SetSearcher(m_kernel, std::move(m_searcher));
    }
}

void ExampleBase::InitKernelDefault(const string &kernelFunctionName, const string &kernelName,
                                 const ktt::DimensionVector &ndRangeDimensions, const vector<ktt::ArgumentId> &arguments)
{
    // Create m_kernel and configure input/output
    m_definition = m_tuner->AddKernelDefinitionFromFile(kernelFunctionName, m_kernelFile, ndRangeDimensions,
        ktt::DimensionVector());
    m_tuner->SetArguments(m_definition, arguments);
        
    m_kernel = m_tuner->CreateSimpleKernel(kernelName, m_definition);
    
}

