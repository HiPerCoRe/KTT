#include <api/computation_result.h>

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

ComputationResult::ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const uint64_t duration,
    const KernelProfilingData& profilingData) :
    status(true),
    duration(duration),
    kernelName(kernelName),
    errorMessage(""),
    configuration(configuration),
    profilingData(profilingData)
{}

ComputationResult::ComputationResult(const std::string& compositionName, const std::vector<ParameterPair>& configuration, const uint64_t duration,
    const std::map<KernelId, KernelProfilingData>& compositionProfilingData) :
    status(true),
    duration(duration),
    kernelName(compositionName),
    errorMessage(""),
    configuration(configuration),
    compositionProfilingData(compositionProfilingData)
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

const std::string& ComputationResult::getKernelName() const
{
    return kernelName;
}

const std::string& ComputationResult::getErrorMessage() const
{
    return errorMessage;
}

const std::vector<ParameterPair>& ComputationResult::getConfiguration() const
{
    return configuration;
}

const KernelProfilingData& ComputationResult::getProfilingData() const
{
    return profilingData;
}

const std::map<KernelId, KernelProfilingData>& ComputationResult::getCompositionProfilingData() const
{
    return compositionProfilingData;
}

} // namespace ktt
