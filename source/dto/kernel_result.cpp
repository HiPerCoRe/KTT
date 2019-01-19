#include <limits>
#include <dto/kernel_result.h>

namespace ktt
{

KernelResult::KernelResult() :
    kernelName(""),
    computationDuration(std::numeric_limits<uint64_t>::max()),
    overhead(0),
    errorMessage(""),
    valid(false)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration) :
    kernelName(kernelName),
    configuration(configuration),
    computationDuration(std::numeric_limits<uint64_t>::max()),
    overhead(0),
    errorMessage(""),
    valid(true)
{}

KernelResult::KernelResult(const std::string& kernelName, uint64_t computationDuration) :
    kernelName(kernelName),
    computationDuration(computationDuration),
    overhead(0),
    errorMessage(""),
    valid(true)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage) :
    kernelName(kernelName),
    configuration(configuration),
    computationDuration(std::numeric_limits<uint64_t>::max()),
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

void KernelResult::setComputationDuration(const uint64_t computationDuration)
{
    this->computationDuration = computationDuration;
}

void KernelResult::setOverhead(const uint64_t overhead)
{
    this->overhead = overhead;
}

void KernelResult::setErrorMessage(const std::string& errorMessage)
{
    this->errorMessage = errorMessage;
}

void KernelResult::setProfilingData(const KernelProfilingData& profilingData)
{
    this->profilingData = profilingData;
}

void KernelResult::setCompositionKernelProfilingData(const KernelId id, const KernelProfilingData& profilingData)
{
    this->compositionProfilingData[id] = profilingData;
}

void KernelResult::setValid(const bool flag)
{
    this->valid = flag;
}

const std::string& KernelResult::getKernelName() const
{
    return kernelName;
}

const KernelConfiguration& KernelResult::getConfiguration() const
{
    return configuration;
}

uint64_t KernelResult::getComputationDuration() const
{
    return computationDuration;
}

uint64_t KernelResult::getOverhead() const
{
    return overhead;
}

const std::string& KernelResult::getErrorMessage() const
{
    return errorMessage;
}

const KernelProfilingData& KernelResult::getProfilingData() const
{
    return profilingData;
}

const KernelProfilingData& KernelResult::getCompositionKernelProfilingData(const KernelId id) const
{
    const auto kernelProfilingData = compositionProfilingData.find(id);
    if (kernelProfilingData == compositionProfilingData.cend())
    {
        throw std::runtime_error(std::string("Profiling data for composition kernel with the following id is not present: ") + std::to_string(id));
    }

    return kernelProfilingData->second;
}

const std::map<KernelId, KernelProfilingData>& KernelResult::getCompositionProfilingData() const
{
    return compositionProfilingData;
}

bool KernelResult::isValid() const
{
    return valid;
}

void KernelResult::increaseOverhead(const uint64_t overhead)
{
    this->overhead += overhead;
}

} // namespace ktt
