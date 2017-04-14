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

std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult)
{
    outputTarget << "Result for kernel <" << tuningResult.kernelName << ">, configuration: " << std::endl;
    outputTarget << tuningResult.configuration;
    outputTarget << "Kernel duration: " << tuningResult.kernelDuration / 1000 << "us" << std::endl;
    outputTarget << "Total duration: " << tuningResult.getTotalDuration() / 1000 << "us" << std::endl;
    return outputTarget;
}

} // namespace ktt
