#include <stdexcept>
#include <api/kernel_profiling_data.h>

namespace ktt
{

KernelProfilingData::KernelProfilingData() :
    remainingProfilingRuns(0),
    validFlag(false)
{}

KernelProfilingData::KernelProfilingData(const uint64_t remainingProfilingRuns) :
    remainingProfilingRuns(remainingProfilingRuns),
    validFlag(false)
{}

KernelProfilingData::KernelProfilingData(const std::vector<KernelProfilingCounter>& profilingCounters) :
    profilingCounters(profilingCounters),
    remainingProfilingRuns(0),
    validFlag(true)
{}

void KernelProfilingData::addCounter(const KernelProfilingCounter& counter)
{
    profilingCounters.push_back(counter);
    validFlag = true;
}

bool KernelProfilingData::hasCounter(const std::string& counterName)
{
    for (const auto& counter : profilingCounters)
    {
        if (counter.getName() == counterName)
        {
            return true;
        }
    }

    return false;
}

const KernelProfilingCounter& KernelProfilingData::getCounter(const std::string& counterName) const
{
    for (const auto& counter : profilingCounters)
    {
        if (counter.getName() == counterName)
        {
            return counter;
        }
    }

    throw std::runtime_error(std::string("Profiling counter with the following name does not exist: ") + counterName);
}

uint64_t KernelProfilingData::getRemainingProfilingRuns() const
{
    return remainingProfilingRuns;
}

const std::vector<KernelProfilingCounter>& KernelProfilingData::getAllCounters() const
{
    return profilingCounters;
}

bool KernelProfilingData::isValid() const
{
    return validFlag;
}

} // namespace ktt
