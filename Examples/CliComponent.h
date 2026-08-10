#pragma once

#include "Ktt.h"
#include <memory>
#include <string>
#include <optional>
#include <functional>
#include <vector>
#include <assert.h>

class CliOption
{
    std::function<void (const std::vector<std::string> &)> m_callback;
    const std::string m_trigger;
    const std::string m_description;
    const std::string m_argumentDescriptions;
    const int m_argumentCount;

public:
    CliOption(std::function<void (const std::vector<std::string> &)> callback, const std::string &trigger, const std::string &description,
              const std::string &argumentDescriptions = "", const int argumentCount = 0);

    std::string get_string() const;

    bool TryTrigger(int argc, char **argv, int &i) const;
};

// struct ExampleConfiguration
// {
//     bool rapidTest = false;
//     bool useProfiling = false;
//     unsigned platform = 0;
//     unsigned device = 0;
//     int problemSize = -1;
//     std::string kernelFile = "";
//     std::unique_ptr<ktt::StopCondition> stopCondition = nullptr;
//     std::unique_ptr<ktt::Searcher> searcher = nullptr;
//     std::string profileSearchModelPath = "";
//     std::optional<ktt::PreciseMeasurementParameters> preciseParams = std::nullopt;
//     bool useDynamicTuning = false;
//     double dynamicTuningTime = 0;
// };

// struct ExampleRefKernelConfiguration : public ExampleConfiguration {
//     std::string refKernelFile = "";
// };

class CliComponent {
public:
    CliComponent();
    void AddOption(const CliOption &option);
    void ProcessInput(int argc, char **argv);

protected:
    std::vector<CliOption> m_options;
};