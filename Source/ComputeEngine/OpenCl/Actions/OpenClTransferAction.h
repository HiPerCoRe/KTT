#pragma once

#ifdef KTT_API_OPENCL

#include <memory>

#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <ComputeEngine/TransferResult.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClTransferAction
{
public:
    OpenClTransferAction(const TransferActionId id, const QueueId queueId, const bool isAsync);

    void SetDuration(const Nanoseconds duration);
    void IncreaseOverhead(const Nanoseconds overhead);
    void SetReleaseFlag();
    void WaitForFinish();

    TransferActionId GetId() const;
    QueueId GetQueueId() const;
    cl_event* GetEvent();
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    bool IsAsync() const;
    TransferResult GenerateResult() const;

private:
    TransferActionId m_Id;
    QueueId m_QueueId;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Duration;
    Nanoseconds m_Overhead;
};

} // namespace ktt

#endif // KTT_API_OPENCL
