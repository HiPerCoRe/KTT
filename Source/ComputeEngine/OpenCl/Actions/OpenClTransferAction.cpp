#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

OpenClTransferAction::OpenClTransferAction(const TransferActionId id) :
    m_Id(id),
    m_Overhead(std::numeric_limits<Nanoseconds>::max())
{
    m_Event = std::make_unique<OpenClEvent>();
}

void OpenClTransferAction::SetOverhead(const Nanoseconds overhead)
{
    m_Overhead = overhead;
}

TransferActionId OpenClTransferAction::GetId() const
{
    return m_Id;
}

OpenClEvent& OpenClTransferAction::GetEvent()
{
    KttAssert(IsValid(), "Only valid transfer actions contain valid event");
    return *m_Event;
}

Nanoseconds OpenClTransferAction::GetOverhead() const
{
    return m_Overhead;
}

bool OpenClTransferAction::IsValid() const
{
    return m_Event != nullptr;
}

} // namespace ktt

#endif // KTT_API_OPENCL
