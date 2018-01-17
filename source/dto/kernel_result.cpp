#include "kernel_result.h"

namespace ktt
{

KernelResult::KernelResult() :
    kernelName(""),
    kernelDuration(UINT64_MAX),
    manipulatorDuration(0),
    overhead(0),
    errorMessage(""),
    valid(false)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration) :
    kernelName(kernelName),
    configuration(configuration),
    kernelDuration(UINT64_MAX),
    manipulatorDuration(0),
    overhead(0),
    errorMessage(""),
    valid(true)
{}

KernelResult::KernelResult(const std::string& kernelName, uint64_t kernelDuration) :
    kernelName(kernelName),
    kernelDuration(kernelDuration),
    manipulatorDuration(0),
    overhead(0),
    errorMessage(""),
    valid(true)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage) :
    kernelName(kernelName),
    configuration(configuration),
    kernelDuration(UINT64_MAX),
    manipulatorDuration(0),
    overhead(0),
    errorMessage(errorMessage),
    valid(false)
{}

void KernelResult::setKernelName(const std::string& kernelName)
{
    this->kernelName = kernelName;
}

void KernelResult::setConfiguration(const KernelConfiguration& configuration)
{
    this->configuration = configuration;
}

void KernelResult::setKernelDuration(const uint64_t kernelDuration)
{
    this->kernelDuration = kernelDuration;
}

void KernelResult::setManipulatorDuration(const uint64_t manipulatorDuration)
{
    this->manipulatorDuration = manipulatorDuration;
}

void KernelResult::setOverhead(const uint64_t overhead)
{
    this->overhead = overhead;
}

void KernelResult::setErrorMessage(const std::string& errorMessage)
{
    this->errorMessage = errorMessage;
}

void KernelResult::setValid(const bool flag)
{
    this->valid = flag;
}

std::string KernelResult::getKernelName() const
{
    return kernelName;
}

KernelConfiguration KernelResult::getConfiguration() const
{
    return configuration;
}

uint64_t KernelResult::getKernelDuration() const
{
    return kernelDuration;
}

uint64_t KernelResult::getManipulatorDuration() const
{
    return manipulatorDuration;
}

uint64_t KernelResult::getOverhead() const
{
    return overhead;
}

uint64_t KernelResult::getTotalDuration() const
{
    return kernelDuration + manipulatorDuration;
}

std::string KernelResult::getErrorMessage() const
{
    return errorMessage;
}

bool KernelResult::isValid() const
{
    return valid;
}

} // namespace ktt
