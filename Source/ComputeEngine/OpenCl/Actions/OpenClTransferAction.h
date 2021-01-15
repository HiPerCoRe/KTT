#pragma once

#ifdef KTT_API_OPENCL

#include <memory>

#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClTransferAction
{
public:
    OpenClTransferAction(const TransferActionId id, const bool isAsync);

    void SetDuration(const Nanoseconds duration);
    void SetOverhead(const Nanoseconds overhead);
    void SetReleaseFlag();
    void WaitForFinish();

    TransferActionId GetId() const;
    cl_event* GetEvent();
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    bool IsAsync() const;

private:
    TransferActionId m_Id;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Duration;
    Nanoseconds m_Overhead;
};

} // namespace ktt

#endif // KTT_API_OPENCL
