#include <Api/Output/KernelResult.h>
#include "KernelResult.h"

namespace ktt
{

KernelResult::KernelResult() :
    m_Timestamp(""),
    m_ExtraDuration(InvalidDuration),
    m_DataMovementOverhead(InvalidDuration),
    m_ValidationOverhead(InvalidDuration),
    m_SearcherOverhead(InvalidDuration),
    m_FailedKernelOverhead(InvalidDuration),
    m_ProfilingRunsOverhead(InvalidDuration),
    m_ProfilingOverhead(InvalidDuration),
    m_ProfilingInfrastructureOverhead(InvalidDuration),
    m_CompilationOverhead(InvalidDuration),
    m_KernelOverhead(InvalidDuration),
    m_KernelOverheadFirstPass(InvalidDuration),
    m_TotalOverhead(InvalidDuration),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& timestamp) :
    m_Configuration(configuration),
    m_KernelName(kernelName),
    m_Timestamp(timestamp),
    m_ExtraDuration(0),
    m_DataMovementOverhead(0),
    m_ValidationOverhead(0),
    m_SearcherOverhead(0),
    m_FailedKernelOverhead(0),
    m_ProfilingRunsOverhead(0),
    m_ProfilingOverhead(0),
    m_ProfilingInfrastructureOverhead(0),
    m_CompilationOverhead(0),
    m_KernelOverhead(0),
    m_KernelOverheadFirstPass(0),
    m_TotalOverhead(0),
    m_Status(ResultStatus::ComputationFailed)
{}

KernelResult::KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
    const std::vector<ComputationResult>& results, const std::string& timestamp) :
    m_Configuration(configuration),
    m_Results(results),
    m_KernelName(kernelName),
    m_Timestamp(timestamp),
    m_ExtraDuration(0),
    m_DataMovementOverhead(0),
    m_ValidationOverhead(0),
    m_SearcherOverhead(0),
    m_FailedKernelOverhead(0),
    m_ProfilingRunsOverhead(0),
    m_ProfilingOverhead(0),
    m_ProfilingInfrastructureOverhead(0),
    m_CompilationOverhead(0),
    m_KernelOverhead(0),
    m_KernelOverheadFirstPass(0),
    m_TotalOverhead(0),
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

void KernelResult::SetProfilingOverhead(const Nanoseconds overhead)
{
    m_ProfilingOverhead = overhead;
}

const std::string& KernelResult::GetKernelName() const
{
    return m_KernelName;
}

const std::vector<ComputationResult>& KernelResult::GetResults() const
{
    return m_Results;
}
const std::string& KernelResult::GetTimestamp() const
{
    return m_Timestamp;
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

Nanoseconds KernelResult::GetKernelOverheadFromCompResults() const
{
    Nanoseconds overhead = 0;

    for (const auto& result : m_Results)
    {
        overhead += result.GetOverhead();
    }

    return overhead;
}

Nanoseconds KernelResult::GetKernelOverhead() const
{
    return m_KernelOverhead;
}

Nanoseconds KernelResult::GetKernelOverheadFirstPass() const
{
    return m_KernelOverheadFirstPass;
}

Nanoseconds KernelResult::GetCompilationOverheadFromCompResults() const
{
    Nanoseconds overhead = 0;

    for (const auto& result : m_Results)
    {
        overhead += result.GetCompilationOverhead();
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

Nanoseconds KernelResult::GetProfilingRunsOverhead() const
{
    return m_ProfilingRunsOverhead;
}

Nanoseconds KernelResult::GetProfilingOverheadFromCompResults() const
{
    Nanoseconds overhead = 0;
    for (const auto &result : m_Results)
    {
        overhead += result.GetProfilingOverhead();
    }
    return overhead;
}

Nanoseconds KernelResult::GetProfilingInfrastructureOverhead() const
{
    return m_ProfilingInfrastructureOverhead;
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
    return m_CompilationOverhead;
}

Nanoseconds KernelResult::GetTotalDuration() const
{
    const Nanoseconds duration = m_ExtraDuration + GetKernelDuration();
    return duration;
}

Nanoseconds KernelResult::GetTotalOverhead() const
{
    if (m_TotalOverhead != InvalidDuration && m_TotalOverhead > 0)
    {
        return m_TotalOverhead;
    }

    Nanoseconds overhead = m_DataMovementOverhead + m_ValidationOverhead + m_SearcherOverhead + m_FailedKernelOverhead;

    if (m_ProfilingRunsOverhead == 0) //without profiling
    {
        overhead += GetKernelOverheadFromCompResults() + GetKernelOverhead(); // kernel overhead of a single pass, that includes compilation overhead
    }
    else //profiling
    { 
        overhead += GetProfilingOverheadFromCompResults() + m_ProfilingRunsOverhead; 
        // GetProfilingInfrastructureOverhead() adds overhead of profiling infrastructure, but without data movement and extra duration
        //profilingRunsOverhead includes kernel overhead and kernel duration of extra passes

    }
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
    m_KernelOverhead += previousResult.GetKernelOverheadFromCompResults();
    m_KernelOverhead += previousResult.GetKernelOverhead();
    m_CompilationOverhead += previousResult.GetCompilationOverheadFromCompResults();
    m_CompilationOverhead += previousResult.GetCompilationOverhead();
    m_ExtraDuration += previousResult.GetExtraDuration();
    m_DataMovementOverhead += previousResult.GetDataMovementOverhead();
    m_ValidationOverhead += previousResult.GetValidationOverhead();
    m_SearcherOverhead += previousResult.GetSearcherOverhead();
    m_FailedKernelOverhead += previousResult.GetFailedKernelOverhead();

    m_ProfilingOverhead += previousResult.GetProfilingOverheadFromCompResults();
    m_ProfilingOverhead += previousResult.GetProfilingOverhead();
    m_ProfilingInfrastructureOverhead += previousResult.GetProfilingOverheadFromCompResults();
    m_ProfilingInfrastructureOverhead += previousResult.GetProfilingInfrastructureOverhead();


    if (first)
    {
        m_KernelOverheadFirstPass = m_KernelOverhead;
        // we want to preserve the original results of the first pass, as these are done without profiling and thus contain the actual duration of the kernel execution
        //moreover, these also contain the actual compilation and kernel overhead that contains actual compilation, not just loading cached version
        // these durations (kernel, compilation and kernel overhead) are then copied into the result of the final pass, after accounting for all the overheads, especially the profiling related ones
        m_Results = previousResult.GetResults();
    }

    if (!first)
    {
        m_ProfilingRunsOverhead += previousResult.GetKernelDuration();
        m_ProfilingRunsOverhead += previousResult.GetKernelOverheadFromCompResults();
        m_ProfilingRunsOverhead += previousResult.GetKernelOverhead();
        m_ProfilingRunsOverhead += previousResult.GetProfilingRunsOverhead();

        m_ProfilingOverhead += previousResult.GetDataMovementOverhead();
        m_ProfilingOverhead += previousResult.GetExtraDuration();
    }

}

void KernelResult::CopyProfilingTimes(const KernelResult& originalResult)
{
    //copy kernel duration, kernel overhead and compilation overhead from the first pass of the kernel into the results of the final pass
    //this ensures that the output files contain the actual kernel duration and compilation overhead
    for (size_t i = 0; i < m_Results.size(); i++) {
        m_Results[i].SetDurationData(
            originalResult.m_Results[i].GetDuration(),
            originalResult.m_Results[i].GetOverhead(),
            originalResult.m_Results[i].GetCompilationOverhead(),
            m_Results[i].GetProfilingOverhead()); // preserve last-pass value
    }
    m_ProfilingRunsOverhead = originalResult.GetProfilingRunsOverhead();
    m_ProfilingOverhead = originalResult.GetProfilingOverhead();
    m_ProfilingInfrastructureOverhead = originalResult.GetProfilingInfrastructureOverhead();
    m_CompilationOverhead = originalResult.GetCompilationOverhead();
    m_KernelOverhead = originalResult.GetKernelOverhead();
    m_KernelOverheadFirstPass = originalResult.GetKernelOverheadFirstPass();
    m_ExtraDuration = originalResult.GetExtraDuration();
    m_DataMovementOverhead = originalResult.GetDataMovementOverhead();
    m_ValidationOverhead = originalResult.GetValidationOverhead();
    m_SearcherOverhead = originalResult.GetSearcherOverhead();
    m_FailedKernelOverhead = originalResult.GetFailedKernelOverhead();
    m_TotalOverhead = originalResult.GetTotalOverhead();
}

void KernelResult::TransferPowerData(const KernelResult& previousResult) 
{
    int size = m_Results.size();
    int prevSize = previousResult.GetResults().size();
    if (size < prevSize)
       for (int i = size; i < prevSize; i++)
           m_Results.push_back(previousResult.GetResults()[i]);
    for (int i = 0; i < prevSize; i++) {
        if (previousResult.GetResults()[i].HasPowerData())
            m_Results[i].SetPowerUsage(
                previousResult.GetResults()[i].GetPowerUsage());
	if (previousResult.GetResults()[i].HasTemperatureData())
            m_Results[i].SetTemperature(
                previousResult.GetResults()[i].GetTemperature());
	if (previousResult.GetResults()[i].HasSMFrequencyData())
            m_Results[i].SetSMFrequency(
                previousResult.GetResults()[i].GetSMFrequency());
        if (previousResult.GetResults()[i].HasMemoryFrequencyData())
            m_Results[i].SetMemoryFrequency(
                previousResult.GetResults()[i].GetMemoryFrequency());
        if (previousResult.GetResults()[i].HasFanSpeedData())
            m_Results[i].SetFanSpeed(
                previousResult.GetResults()[i].GetFanSpeed());
    }
}

} // namespace ktt
