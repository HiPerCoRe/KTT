#pragma once

#include "tuning_result.h"

namespace ktt
{

TuningResult::TuningResult(const uint64_t duration, const KernelConfiguration& configuration):
    duration(duration),
    configuration(configuration)
{}

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
    outputTarget << "Printing tuning result for configuration: " << tuningResult.configuration << std::endl;
    outputTarget << "Kernel execution duration: " << tuningResult.duration << std::endl;
    return outputTarget;
}

} // namespace ktt
