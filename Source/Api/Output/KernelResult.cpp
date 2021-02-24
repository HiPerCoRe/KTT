#include <Api/Output/KernelResult.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_ExtraDuration(InvalidDuration),
    m_ExtraOverhead(InvalidDuration),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration) :
    m_Configuration(configuration),
    m_KernelName(kernelName),
    m_ExtraDuration(0),
    m_ExtraOverhead(0),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
    const std::vector<ComputationResult>& results) :
    m_Configuration(configuration),
    m_Results(results),
    m_KernelName(kernelName),
    m_ExtraDuration(0),
    m_ExtraOverhead(0),
    m_Status(ResultStatus::Ok)
{}

void KernelResult::SetStatus(const ResultStatus status)
{
    KttAssert(status != ResultStatus::Ok, "Status Ok should be set only by calling the constructor with computation results");
    m_Status = status;
}

void KernelResult::SetExtraDuration(const Nanoseconds duration)
{
    m_ExtraDuration = duration;
}

void KernelResult::SetExtraOverhead(const Nanoseconds overhead)
{
    m_ExtraOverhead = overhead;
}

const std::string& KernelResult::GetKernelName() const
{
    return m_KernelName;
}

const std::vector<ComputationResult>& KernelResult::GetResults() const
{
    return m_Results;
}

const KernelConfiguration& KernelResult::GetConfiguration() const
{
    return m_Configuration;
}

ResultStatus KernelResult::GetStatus() const
{
    return m_Status;
}

Nanoseconds KernelResult::GetKernelDuration() const
{
    Nanoseconds duration = 0;

    for (const auto& result : m_Results)
    {
        duration += result.GetDuration();
    }

    return duration;
}

Nanoseconds KernelResult::GetKernelOverhead() const
{
    Nanoseconds overhead = 0;

    for (const auto& result : m_Results)
    {
        overhead += result.GetOverhead();
    }

    return overhead;
}

Nanoseconds KernelResult::GetExtraDuration() const
{
    return m_ExtraDuration;
}

Nanoseconds KernelResult::GetExtraOverhead() const
{
    return m_ExtraOverhead;
}

Nanoseconds KernelResult::GetTotalDuration() const
{
    const Nanoseconds duration = m_ExtraDuration + GetKernelDuration();
    return duration;
}

Nanoseconds KernelResult::GetTotalOverhead() const
{
    const Nanoseconds overhead = m_ExtraOverhead + GetKernelOverhead();
    return overhead;
}

bool KernelResult::IsValid() const
{
    return m_Status == ResultStatus::Ok;
}

bool KernelResult::HasRemainingProfilingRuns() const
{
    for (const auto& result : m_Results)
    {
        if (result.HasRemainingProfilingRuns())
        {
            return true;
        }
    }

    return false;
}

} // namespace ktt
