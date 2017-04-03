#pragma once

#include "tuning_result.h"

namespace ktt
{

TuningResult::TuningResult(const std::string& kernelName, const uint64_t duration, const KernelConfiguration& configuration):
    kernelName(kernelName),
    duration(duration),
    configuration(configuration)
{}

std::string TuningResult::getKernelName() const
{
    return kernelName;
}

uint64_t TuningResult::getDuration() const
{
    return duration;
}

KernelConfiguration TuningResult::getConfiguration() const
{
    return configuration;
}

std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult)
{
    outputTarget << "Printing tuning result for kernel <" << tuningResult.kernelName << "> with configuration: " << std::endl;
    outputTarget << tuningResult.configuration;
    outputTarget << "Kernel execution duration: " << tuningResult.duration / 1000 << "us" << std::endl;
    return outputTarget;
}

} // namespace ktt
