#include <Api/Output/KernelResult.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_ExtraDuration(InvalidDuration),
    m_DataMovementOverhead(InvalidDuration),
    m_ValidationOverhead(InvalidDuration),
    m_SearcherOverhead(InvalidDuration),
    m_FailedKernelOverhead(InvalidDuration),
    m_ProfilingRunsOverhead(InvalidDuration),
    m_ProfilingOverhead(InvalidDuration),
    m_CompilationOverhead(InvalidDuration),
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
    m_ProfilingRunsOverhead(0),
    m_ProfilingOverhead(0),
    m_CompilationOverhead(0),
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
    m_ProfilingRunsOverhead(0),
    m_ProfilingOverhead(0),
    m_CompilationOverhead(0),
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

void KernelResult::SetProfilingRunsOverhead(const Nanoseconds overhead)
{
    m_ProfilingRunsOverhead = overhead;
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

Nanoseconds KernelResult::GetKernelCompilationOverhead() const
{
    Nanoseconds overhead = 0;

    for (const auto& result : m_Results)
    {
        overhead += result.GetCompilationOverhead();
    }

    return overhead + m_CompilationOverhead;
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

Nanoseconds KernelResult::GetProfilingRunsOverhead() const
{
    return m_ProfilingRunsOverhead;
}

Nanoseconds KernelResult::GetProfilingOverhead() const
{
    return m_ProfilingOverhead;
}

Nanoseconds KernelResult::GetProfilingTotalOverhead() const
{
    return m_ProfilingOverhead + m_ProfilingRunsOverhead;
}

Nanoseconds KernelResult::GetCompilationOverhead() const
{
    return GetKernelCompilationOverhead();
}

Nanoseconds KernelResult::GetTotalDuration() const
{
    const Nanoseconds duration = m_ExtraDuration + GetKernelDuration();
    return duration;
}

Nanoseconds KernelResult::GetTotalOverhead() const
{
    Nanoseconds overhead = m_DataMovementOverhead + m_ValidationOverhead + m_SearcherOverhead + /*GetKernelOverhead() +*/ m_FailedKernelOverhead + m_ProfilingRunsOverhead + m_CompilationOverhead;
    if (m_ProfilingRunsOverhead == 0)
        overhead += GetKernelOverhead(); //in case there is no profiling, include also actual kernel overhead (was not fused)
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

void KernelResult::FuseProfilingTimes(const KernelResult& previousResult, bool first)
{
    if (! first) {
        m_ProfilingRunsOverhead += previousResult.GetKernelDuration();
        m_ProfilingRunsOverhead += previousResult.GetKernelOverhead();
        m_ProfilingRunsOverhead += previousResult.GetProfilingRunsOverhead();
        m_ProfilingOverhead += previousResult.GetDataMovementOverhead();
        m_ProfilingOverhead += previousResult.GetExtraDuration();
        m_ProfilingOverhead += previousResult.GetProfilingOverhead();
    }
    m_CompilationOverhead += previousResult.GetCompilationOverhead();
    m_ExtraDuration += previousResult.GetExtraDuration();
    m_DataMovementOverhead += previousResult.GetDataMovementOverhead();
    m_ValidationOverhead += previousResult.GetValidationOverhead();
    m_SearcherOverhead += previousResult.GetSearcherOverhead();
    m_FailedKernelOverhead += previousResult.GetFailedKernelOverhead();
}

void KernelResult::CopyProfilingTimes(const KernelResult& originalResult)
{
    m_ProfilingRunsOverhead = originalResult.GetProfilingRunsOverhead();
    m_ProfilingOverhead = originalResult.GetProfilingOverhead();
    m_CompilationOverhead = originalResult.GetCompilationOverhead();
    m_ExtraDuration = originalResult.GetExtraDuration();
    m_DataMovementOverhead = originalResult.GetDataMovementOverhead();
    m_ValidationOverhead = originalResult.GetValidationOverhead();
    m_SearcherOverhead = originalResult.GetSearcherOverhead();
    m_FailedKernelOverhead = originalResult.GetFailedKernelOverhead();
}


} // namespace ktt
