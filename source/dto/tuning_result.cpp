#include "tuning_result.h"

namespace ktt
{

TuningResult::TuningResult(const std::string& kernelName, const KernelConfiguration& configuration) :
    kernelName(kernelName),
    configuration(configuration),
    kernelDuration(UINT64_MAX),
    kernelOverhead(0),
    manipulatorDuration(0),
    valid(false),
    statusMessage("Ok")
{}

TuningResult::TuningResult(const std::string& kernelName, const KernelConfiguration& configuration, const KernelRunResult& kernelRunResult) :
    kernelName(kernelName),
    configuration(configuration),
    kernelDuration(kernelRunResult.getDuration()),
    kernelOverhead(kernelRunResult.getOverhead()),
    manipulatorDuration(0),
    valid(kernelRunResult.isValid()),
    statusMessage("Ok")
{}

TuningResult::TuningResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& statusMessage) :
    kernelName(kernelName),
    configuration(configuration),
    kernelDuration(UINT64_MAX),
    kernelOverhead(0),
    manipulatorDuration(0),
    valid(false),
    statusMessage(statusMessage)
{}

void TuningResult::setKernelDuration(const uint64_t kernelDuration)
{
    this->kernelDuration = kernelDuration;
}

void TuningResult::setKernelOverhead(const uint64_t kernelOverhead)
{
    this->kernelOverhead = kernelOverhead;
}

void TuningResult::setManipulatorDuration(const uint64_t manipulatorDuration)
{
    this->manipulatorDuration = manipulatorDuration;
}

void TuningResult::setValid(const bool flag)
{
    this->valid = flag;
}

void TuningResult::setStatusMessage(const std::string& statusMessage)
{
    this->statusMessage = statusMessage;
}

std::string TuningResult::getKernelName() const
{
    return kernelName;
}

KernelConfiguration TuningResult::getConfiguration() const
{
    return configuration;
}

uint64_t TuningResult::getKernelDuration() const
{
    return kernelDuration;
}

uint64_t TuningResult::getKernelOverhead() const
{
    return kernelOverhead;
}

uint64_t TuningResult::getManipulatorDuration() const
{
    return manipulatorDuration;
}

uint64_t TuningResult::getTotalDuration() const
{
    return kernelDuration + manipulatorDuration;
}

bool TuningResult::isValid() const
{
    return valid;
}

std::string TuningResult::getStatusMessage() const
{
    return statusMessage;
}

} // namespace ktt
