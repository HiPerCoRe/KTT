#include <Api/Output/KernelProfilingData.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelProfilingData::KernelProfilingData(const uint64_t remainingRuns) :
    m_RemainingRuns(remainingRuns)
{}

KernelProfilingData::KernelProfilingData(const std::vector<KernelProfilingCounter>& counters) :
    m_RemainingRuns(0)
{
    SetCounters(counters);
}

bool KernelProfilingData::IsValid() const
{
    return m_RemainingRuns == 0;
}

bool KernelProfilingData::HasCounter(const std::string& name) const
{
    return ContainsElementIf(m_Counters, [&name](const auto& counter)
    {
        return name == counter.GetName();
    });
}

const KernelProfilingCounter& KernelProfilingData::GetCounter(const std::string& name) const
{
    for (const auto& counter : m_Counters)
    {
        if (counter.GetName() == name)
        {
            return counter;
        }
    }

    throw KttException("Profiling counter with the following name was not found: " + name);
}

const std::vector<KernelProfilingCounter>& KernelProfilingData::GetCounters() const
{
    return m_Counters;
}

void KernelProfilingData::SetCounters(const std::vector<KernelProfilingCounter>& counters)
{
    KttAssert(ContainsUniqueElements(counters), "Counters with duplicit name were found");
    m_Counters = counters;
    m_RemainingRuns = 0;
}

void KernelProfilingData::AddCounter(const KernelProfilingCounter& counter)
{
    m_Counters.push_back(counter);
    KttAssert(ContainsUniqueElements(m_Counters), "Counter with duplicit name was found");
    m_RemainingRuns = 0;
}

bool KernelProfilingData::HasRemainingProfilingRuns() const
{
    return m_RemainingRuns > 0;
}

uint64_t KernelProfilingData::GetRemainingProfilingRuns() const
{
    return m_RemainingRuns;
}

void KernelProfilingData::DecreaseRemainingProfilingRuns()
{
    if (m_RemainingRuns > 0)
    {
        --m_RemainingRuns;
    }
}

} // namespace ktt
