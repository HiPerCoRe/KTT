#include "ExampleConfigurator.h"
#include "Api/Configuration/PreciseMeasurementParameters.h"
#include <functional>
#include <vector>
#include <assert.h>
#include <iostream>

using namespace std;

CliOption::CliOption(function<void (const vector<string> &)> callback, const string &trigger, const string &description,
              const string &argumentDescriptions, const int argumentCount)
        : m_callback(callback), m_trigger(trigger), m_description(description),
          m_argumentDescriptions(argumentDescriptions), m_argumentCount(argumentCount)
{
}

string CliOption::get_string() const
{
    return m_trigger + " " + m_argumentDescriptions + "\n\t" + m_description;
}

bool CliOption::TryTrigger(int argc, char **argv, int &i) const {
    assert(i < argc);
    if (argv[i] != m_trigger) return false;
    if (i + m_argumentCount >= argc)
    {
        cerr << m_trigger << " expects a value to be passed!" << endl;
        exit(1);
    }
    vector<string> arguments;
    for (int j = 0; j < m_argumentCount; ++j) {
        arguments.push_back(argv[++i]);
    }
    m_callback(arguments);
    return true;
}

void SetUpCommonOptions(vector<CliOption> &options, ExampleConfiguration *config) {
    options.emplace_back([&options](const vector<string> &) {
        cout << "Usage: program [options]" << endl << endl;
        cout << "Options:" << endl;
        for (const auto& option : options) {
            cout << option.get_string() << endl;
        }
        exit(0);
    }, "--help", "Show this help message and exit.");

    options.emplace_back([config](const vector<string> &) {
        config->rapidTest = true;
    }, "--rapidTest", "Run in rapid test mode");

    options.emplace_back([config](const vector<string> &) {
        config->useProfiling = true;
    }, "--profile", "Enable profiling");

    options.emplace_back([config](const vector<string> &args) {
        config->platform = stoul(args[0]);
    }, "--platform", "Platform index (expects int)", "<index>", 1);

    options.emplace_back([config](const vector<string> &args) {
        config->device = stoul(args[0]);
    }, "--device", "Device index (expects int)", "<index>", 1);

    options.emplace_back([config](const vector<string> &args) {
        config->problemSize = stoi(args[0]);
    }, "--problemSize", "Problem size in MiB (expects int)", "<size>", 1);

    options.emplace_back([config](const vector<string> &args) {
        config->kernelFile = args[0];
    }, "--kernelPath", "Kernel file path (expects string)", "<path>", 1);

    options.emplace_back([config](const vector<string> &args) {
        if (args[0] == "ds") {
            config->searcher = make_unique<ktt::DeterministicSearcher>();
        } else if (args[0] == "random") {
            config->searcher = make_unique<ktt::RandomSearcher>();
        } else if (args[0] == "mcmc") {
            config->searcher = make_unique<ktt::McmcSearcher>();
        } else {
            cerr << "--searcher expects one of (ds, random, mcmc)\n";
            exit(1);
        }
    }, "--searcher", "Searcher type (ds, random, mcmc)", "<type>", 1);

    options.emplace_back([config](const vector<string> &args) {
        config->profileSearchModelPath = args[0];
        config->useProfiling = true;
    }, "--profileSearcher", 
    "Enable profile searcher and set path to model (expects string) (functions only on CUDA devices)",
    "<pathToModel>", 1);

    options.emplace_back([config](const vector<string> &args) {
        if (args[0] == "confs") {
            config->stopCondition = make_unique<ktt::ConfigurationCount>(stoul(args[1]));
        } else if (args[0] == "fails") {
            config->stopCondition = make_unique<ktt::FailureCount>(stoul(args[1]));
        } else if (args[0] == "time") {
            config->stopCondition = make_unique<ktt::TuningDuration>(stod(args[1]));
        } else if (args[0] == "best") {
            config->stopCondition = make_unique<ktt::ConfigurationDuration>(stod(args[1]));
        } else {
            cerr << "--stopCondition expects one of (confs, fails, time, best)\n";
            exit(1);
        }
    }, "--stopCondition", 
    "Set a stop condition. <type> can be confs, fails, time, best. "
    "<limit> is respectively configuration count (ulong), failed kernel run count (ulong), "
    "total tuning duration in seconds (double), best configuration duration in milliseconds (double).",
    "<type> <limit>", 2);

    options.emplace_back([config](const vector<string> &args) {
        config->preciseParams = ktt::PreciseMeasurementParameters(stoul(args[0]),
            stoul(args[1]), stod(args[2]));
    }, "--preciseParams", "Set PreciseMeasurementParameters, calculationDurationMethod is the default Minimum, refer to KTT documentation for details.",
    "<minTimeMs> <maxTimeMs> <maxPowerDiff>", 3);
    options.emplace_back([config](const vector<string> &args) {
        if (config->preciseParams == std::nullopt) {
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
        config->preciseParams->durationCalculationMethod = calcMethod;
    }, "--preciseParamsCalcMethod", "Optionally set PreciseMeasurementParameters::durationCalculationMethod AFTER USING --preciseParams, expects one of "
    "(min, median, avg), refer to KTT documentation for details.",
    "<calcMethod>", 1);

    options.emplace_back([config](const vector<string> &args) {
        config->useDynamicTuning = true;
        config->dynamicTuningTime = stod(args[0]);
    }, "--useDynamicTuning", "Enables a basic implementation of dynamic tuning."
    "The tuning will last <time> (double) seconds and then the only the best configuration will be run.",
    "<time>", 1);
}

void SetUpRefKernelOption(vector<CliOption> &options, ExampleRefKernelConfiguration &config) {
    options.emplace_back([&config](const vector<string> &args) {
        config.refKernelFile = args[0];
    }, "--refKernelPath", "Reference kernel file path (expects string)", "<path>", 1);
}

void IterateArguments(int argc, char **argv, const vector<CliOption> &options) {
    for (int i = 1; i < argc; ++i) {
        bool triggered = false;
        for (const auto& option : options) {
            if (option.TryTrigger(argc, argv, i)) {
                triggered = true;
                break;
            }
        }
        if (!triggered) {
            cerr << argv[i] << " is not a valid option.\n";
            exit(1);
        }
    }
}

ExampleConfiguration ProcessInput(int argc, char **argv) {
    ExampleConfiguration config;
    vector<CliOption> options;
    SetUpCommonOptions(options, &config);

    IterateArguments(argc, argv, options);

    return config;
}

ExampleRefKernelConfiguration RefKernelProcessInput(int argc, char **argv) {
    ExampleRefKernelConfiguration config;
    vector<CliOption> options;
    SetUpCommonOptions(options, &config);
    SetUpRefKernelOption(options, config);

    IterateArguments(argc, argv, options);

    return config;
}
