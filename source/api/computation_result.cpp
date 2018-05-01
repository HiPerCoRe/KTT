#include "computation_result.h"

namespace ktt
{

ComputationResult::ComputationResult() :
    status(false),
    duration(UINT64_MAX),
    kernelName(""),
    errorMessage("")
{}

ComputationResult::ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const uint64_t duration) :
    status(true),
    duration(duration),
    kernelName(kernelName),
    errorMessage(""),
    configuration(configuration)
{}

ComputationResult::ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration,
    const std::string& errorMessage) :
    status(false),
    duration(UINT64_MAX),
    kernelName(kernelName),
    errorMessage(errorMessage),
    configuration(configuration)
{}

bool ComputationResult::getStatus() const
{
    return status;
}

uint64_t ComputationResult::getDuration() const
{
    return duration;
}

std::string ComputationResult::getKernelName() const
{
    return kernelName;
}

std::string ComputationResult::getErrorMessage() const
{
    return errorMessage;
}

std::vector<ParameterPair> ComputationResult::getConfiguration() const
{
    return configuration;
}

} // namespace ktt
