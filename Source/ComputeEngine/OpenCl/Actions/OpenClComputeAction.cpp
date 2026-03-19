#ifdef KTT_API_OPENCL

#include <string>

#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClComputeAction::OpenClComputeAction(const ComputeActionId id, const QueueId queueId, std::shared_ptr<OpenClKernel> kernel,
    const DimensionVector& globalSize, const DimensionVector& localSize) :
    m_Id(id),
    m_QueueId(queueId),
    m_Kernel(kernel),
    m_Overhead(0),
    m_CompilationOverhead(0),
    m_ProfilingOverhead(0),
    m_GlobalSize(globalSize),
    m_LocalSize(localSize),
    m_DurationFromMultirun(std::nullopt),
    m_DurationStdev(std::nullopt)
{
    Logger::LogDebug("Initializing OpenCL compute action with id " + std::to_string(id)
        + " for kernel with name " + kernel->GetName());
    KttAssert(m_Kernel != nullptr, "Invalid kernel was used during OpenCL compute action initialization");

    m_Event = std::make_unique<OpenClEvent>();
}

void OpenClComputeAction::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void OpenClComputeAction::IncreaseCompilationOverhead(const Nanoseconds overhead)
{
    m_CompilationOverhead += overhead;
}

void OpenClComputeAction::IncreaseProfilingOverhead(const Nanoseconds overhead)
{
    m_ProfilingOverhead += overhead;
}

void OpenClComputeAction::SetComputeId(const KernelComputeId& id)
{
    m_ComputeId = id;
}

void OpenClComputeAction::SetReleaseFlag()
{
    m_Event->SetReleaseFlag();
}

void OpenClComputeAction::SetDurationFromMultirun(const Nanoseconds duration)
{
    m_DurationFromMultirun = duration;
}

void OpenClComputeAction::SetDurationStdev(const double durationStdev)
{
    m_DurationStdev = durationStdev;
}

void OpenClComputeAction::WaitForFinish()
{
    Logger::LogDebug("Waiting for OpenCL kernel compute action with id " + std::to_string(m_Id));
    m_Event->WaitForFinish();
}

ComputeActionId OpenClComputeAction::GetId() const
{
    return m_Id;
}

QueueId OpenClComputeAction::GetQueueId() const
{
    return m_QueueId;
}

OpenClKernel& OpenClComputeAction::GetKernel()
{
    return *m_Kernel;
}

cl_event* OpenClComputeAction::GetEvent()
{
    return m_Event->GetEvent();
}

Nanoseconds OpenClComputeAction::GetDuration() const
{
    return m_Event->GetDuration();
}

Nanoseconds OpenClComputeAction::GetOverhead() const
{
    return m_Overhead;
}

Nanoseconds OpenClComputeAction::GetCompilationOverhead() const
{
    return m_CompilationOverhead;
}

Nanoseconds OpenClComputeAction::GetProfilingOverhead() const
{    
    return m_ProfilingOverhead;
}

const KernelComputeId& OpenClComputeAction::GetComputeId() const
{
    return m_ComputeId;
}

ComputationResult OpenClComputeAction::GenerateResult() const
{
    ComputationResult result(m_Kernel->GetName());
    Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();
    const Nanoseconds compilationOverhead = GetCompilationOverhead();
    const Nanoseconds profilingOverhead = GetProfilingOverhead();
    std::unique_ptr<KernelCompilationData> compilationData = m_Kernel->GenerateCompilationData();

    // Use duration from multirun if available
    if (m_DurationFromMultirun.has_value()) {
        duration = m_DurationFromMultirun.value();
    }

    result.SetDurationData(duration, overhead, compilationOverhead, profilingOverhead);
    result.SetSizeData(m_GlobalSize, m_LocalSize);
    result.SetCompilationData(std::move(compilationData));

    // Set duration standard deviation if available
    if (m_DurationStdev.has_value()) {
        result.SetDurationStdev(m_DurationStdev.value());
    }

    return result;
}

} // namespace ktt

#endif // KTT_API_OPENCL
