#include "tuning_result.h"

namespace ktt
{

TuningResult::TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const KernelConfiguration& configuration) :
    kernelName(kernelName),
    kernelDuration(kernelDuration),
    manipulatorDuration(0),
    configuration(configuration)
{}

TuningResult::TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const uint64_t manipulatorDuration,
    const KernelConfiguration& configuration) :
    kernelName(kernelName),
    kernelDuration(kernelDuration),
    manipulatorDuration(manipulatorDuration),
    configuration(configuration)
{}

std::string TuningResult::getKernelName() const
{
    return kernelName;
}

uint64_t TuningResult::getKernelDuration() const
{
    return kernelDuration;
}

uint64_t TuningResult::getManipulatorDuration() const
{
    return manipulatorDuration;
}

uint64_t TuningResult::getTotalDuration() const
{
    return kernelDuration + manipulatorDuration;
}

KernelConfiguration TuningResult::getConfiguration() const
{
    return configuration;
}

} // namespace ktt
