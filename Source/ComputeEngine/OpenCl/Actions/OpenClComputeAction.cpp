#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

OpenClComputeAction::OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel) :
    m_Id(id),
    m_Kernel(kernel),
    m_Overhead(InvalidDuration)
{
    m_Event = std::make_unique<OpenClEvent>();
}

void OpenClComputeAction::SetOverhead(const Nanoseconds overhead)
{
    m_Overhead = overhead;
}

void OpenClComputeAction::SetReleaseFlag()
{
    m_Event->SetReleaseFlag();
}

void OpenClComputeAction::WaitForFinish()
{
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

} // namespace ktt

#endif // KTT_API_OPENCL
