#ifdef KTT_API_OPENCL

#include <string>

#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClComputeAction::OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel,
    const DimensionVector& globalSize, const DimensionVector& localSize) :
    m_Id(id),
    m_Kernel(kernel),
    m_Overhead(0),
    m_GlobalSize(globalSize),
    m_LocalSize(localSize)
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

void OpenClComputeAction::SetConfigurationPrefix(const std::string& prefix)
{
    m_Prefix = prefix;
}

void OpenClComputeAction::SetReleaseFlag()
{
    m_Event->SetReleaseFlag();
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

const std::string& OpenClComputeAction::GetConfigurationPrefix() const
{
    return m_Prefix;
}

KernelComputeId OpenClComputeAction::GetComputeId() const
{
    return m_Kernel->GetName() + m_Prefix;
}

ComputationResult OpenClComputeAction::GenerateResult() const
{
    ComputationResult result(m_Kernel->GetName());
    const Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();
    std::unique_ptr<KernelCompilationData> compilationData = m_Kernel->GenerateCompilationData();

    result.SetDurationData(duration, overhead);
    result.SetSizeData(m_GlobalSize, m_LocalSize);
    result.SetCompilationData(std::move(compilationData));

    return result;
}

} // namespace ktt

#endif // KTT_API_OPENCL
