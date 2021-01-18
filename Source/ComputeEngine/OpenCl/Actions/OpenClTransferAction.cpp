#ifdef KTT_API_OPENCL

#include <string>

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClTransferAction::OpenClTransferAction(const TransferActionId id, const bool isAsync) :
    m_Id(id),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{
    Logger::LogDebug(std::string("Initializing OpenCL transfer action with id ") + std::to_string(id));

    if (isAsync)
    {
        m_Event = std::make_unique<OpenClEvent>();
    }
}

void OpenClTransferAction::SetDuration(const Nanoseconds duration)
{
    KttAssert(!IsAsync(), "Duration for async actions is handled by events");
    m_Duration = duration;
}

void OpenClTransferAction::SetOverhead(const Nanoseconds overhead)
{
    m_Overhead = overhead;
}

void OpenClTransferAction::SetReleaseFlag()
{
    KttAssert(IsAsync(), "Only async actions contain valid event");
    m_Event->SetReleaseFlag();
}

void OpenClTransferAction::WaitForFinish()
{
    Logger::LogDebug(std::string("Waiting for OpenCL transfer action with id ") + std::to_string(m_Id));

    if (IsAsync())
    {
        m_Event->WaitForFinish();
    }
}

TransferActionId OpenClTransferAction::GetId() const
{
    return m_Id;
}

cl_event* OpenClTransferAction::GetEvent()
{
    KttAssert(IsAsync(), "Only async actions contain valid event");
    return m_Event->GetEvent();
}

Nanoseconds OpenClTransferAction::GetDuration() const
{
    if (IsAsync())
    {
        return m_Event->GetDuration();
    }

    return m_Duration;
}

Nanoseconds OpenClTransferAction::GetOverhead() const
{
    return m_Overhead;
}

bool OpenClTransferAction::IsAsync() const
{
    return m_Event != nullptr;
}

} // namespace ktt

#endif // KTT_API_OPENCL
