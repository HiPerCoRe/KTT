#pragma once

#include "Ktt.h"
#include <memory>
#include <string>
#include <optional>

struct ExampleConfiguration
{
    bool rapidTest;
    bool useProfiling;
    unsigned platform;
    unsigned device;
    int problemSize = -1;
    std::string kernelFile;
    std::unique_ptr<ktt::StopCondition> stopCondition;
    std::unique_ptr<ktt::Searcher> searcher;
    std::string profileSearchModelPath;
    std::optional<ktt::PreciseMeasurementParameters> preciseParams;
    bool useDynamicTuning;
};

struct ExampleRefKernelConfiguration : public ExampleConfiguration {
    std::string refKernelFile;
};

ExampleConfiguration ProcessInput(int argc, char **argv);

ExampleRefKernelConfiguration RefKernelProcessInput(int argc, char **argv);