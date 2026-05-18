#pragma once

#include "Ktt.h"
#include <memory>
#include <string>
#include <optional>

struct ExampleConfiguration
{
    bool rapidTest = false;
    bool useProfiling = false;
    unsigned platform = 0;
    unsigned device = 0;
    int problemSize = -1;
    std::string kernelFile = "";
    std::unique_ptr<ktt::StopCondition> stopCondition = nullptr;
    std::unique_ptr<ktt::Searcher> searcher = nullptr;
    std::string profileSearchModelPath = "";
    std::optional<ktt::PreciseMeasurementParameters> preciseParams = std::nullopt;
    bool useDynamicTuning = false;
    double dynamicTuningTime = 0;
};

struct ExampleRefKernelConfiguration : public ExampleConfiguration {
    std::string refKernelFile = "";
};

ExampleConfiguration ProcessInput(int argc, char **argv);

ExampleRefKernelConfiguration RefKernelProcessInput(int argc, char **argv);