#include <Api/Output/KernelResult.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_ExtraDuration(InvalidDuration),
    m_DataMovementOverhead(InvalidDuration),
    m_ValidationOverhead(InvalidDuration),
    m_SearcherOverhead(InvalidDuration),
    m_FailedKernelOverhead(InvalidDuration),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration) :
    m_Configuration(configuration),
    m_KernelName(kernelName),
    m_ExtraDuration(0),
    m_DataMovementOverhead(0),
    m_ValidationOverhead(0),
    m_SearcherOverhead(0),
    m_FailedKernelOverhead(0),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
    const std::vector<ComputationResult>& results) :
    m_Configuration(configuration),
    m_Results(results),
    m_KernelName(kernelName),
    m_ExtraDuration(0),
    m_DataMovementOverhead(0),
    m_ValidationOverhead(0),
    m_SearcherOverhead(0),
    m_FailedKernelOverhead(0),
    m_Status(ResultStatus::Ok)
{}

void KernelResult::SetStatus(const ResultStatus status)
{
    m_Status = status;
}

void KernelResult::SetExtraDuration(const Nanoseconds duration)
{
    m_ExtraDuration = duration;
}

void KernelResult::SetExtraOverhead(const Nanoseconds overhead)
{
    m_DataMovementOverhead = overhead;
}

void KernelResult::SetDataMovementOverhead(const Nanoseconds overhead)
{
    m_DataMovementOverhead = overhead;
}

void KernelResult::SetValidationOverhead(const Nanoseconds overhead)
{
    m_ValidationOverhead = overhead;
}

void KernelResult::SetSearcherOverhead(const Nanoseconds overhead)
{
    m_SearcherOverhead = overhead;
}

void KernelResult::SetFailedKernelOverhead(const Nanoseconds overhead)
{
    m_FailedKernelOverhead = overhead;
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
    return m_DataMovementOverhead;
}

Nanoseconds KernelResult::GetDataMovementOverhead() const
{
    return m_DataMovementOverhead;
}

Nanoseconds KernelResult::GetValidationOverhead() const
{
    return m_ValidationOverhead;
}

Nanoseconds KernelResult::GetSearcherOverhead() const
{
    return m_SearcherOverhead;
}

Nanoseconds KernelResult::GetFailedKernelOverhead() const
{
    return m_FailedKernelOverhead;
}

Nanoseconds KernelResult::GetTotalDuration() const
{
    const Nanoseconds duration = m_ExtraDuration + GetKernelDuration();
    return duration;
}

Nanoseconds KernelResult::GetTotalOverhead() const
{
    const Nanoseconds overhead = m_DataMovementOverhead + m_ValidationOverhead + m_SearcherOverhead + GetKernelOverhead() + m_FailedKernelOverhead;
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
