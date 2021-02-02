#include <Api/Output/KernelResult.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_Id(InvalidKernelId),
    m_ExtraDuration(InvalidDuration),
    m_ExtraOverhead(0)
{}

KernelResult::KernelResult(const KernelId id, const std::vector<ComputationResult>& results) :
    m_Results(results),
    m_Id(id),
    m_ExtraDuration(0),
    m_ExtraOverhead(0)
{}

void KernelResult::SetExtraDuration(const Nanoseconds duration)
{
    m_ExtraDuration = duration;
}

void KernelResult::SetExtraOverhead(const Nanoseconds overhead)
{
    m_ExtraOverhead = overhead;
}

KernelId KernelResult::GetId() const
{
    return m_Id;
}

const std::vector<ComputationResult>& KernelResult::GetResults() const
{
    return m_Results;
}

Nanoseconds KernelResult::GetTotalDuration() const
{
    Nanoseconds duration = m_ExtraDuration;

    for (const auto& result : m_Results)
    {
        duration += result.GetDuration();
    }

    return duration;
}

Nanoseconds KernelResult::GetTotalOverhead() const
{
    Nanoseconds overhead = m_ExtraOverhead;

    for (const auto& result : m_Results)
    {
        overhead += result.GetOverhead();
    }

    return overhead;
}

bool KernelResult::IsValid() const
{
    return m_Id != InvalidKernelId;
}

} // namespace ktt
