#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

OpenClComputeAction::OpenClComputeAction(const ComputeActionId id) :
    m_Id(id),
    m_Overhead(std::numeric_limits<Nanoseconds>::max())
{}

OpenClComputeAction::OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel) :
    m_Id(id),
    m_Kernel(kernel),
    m_Overhead(std::numeric_limits<Nanoseconds>::max())
{
    m_Event = std::make_unique<OpenClEvent>();
}

void OpenClComputeAction::SetOverhead(const Nanoseconds overhead)
{
    m_Overhead = overhead;
}

ComputeActionId OpenClComputeAction::GetId() const
{
    return m_Id;
}

OpenClKernel& OpenClComputeAction::GetKernel()
{
    KttAssert(IsValid(), "Only valid compute actions contain valid kernel");
    return *m_Kernel;
}

OpenClEvent& OpenClComputeAction::GetEvent()
{
    KttAssert(IsValid(), "Only valid compute actions contain valid event");
    return *m_Event;
}

Nanoseconds OpenClComputeAction::GetOverhead() const
{
    return m_Overhead;
}

bool OpenClComputeAction::IsValid() const
{
    return m_Event != nullptr;
}

} // namespace ktt

#endif // KTT_API_OPENCL
