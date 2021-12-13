#ifdef KTT_API_CUDA

#include <string>

#include <ComputeEngine/Cuda/Actions/CudaTransferAction.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaTransferAction::CudaTransferAction(const TransferActionId id, const QueueId queueId) :
    m_Id(id),
    m_QueueId(queueId),
    m_Overhead(0)
{
    Logger::LogDebug("Initializing CUDA transfer action with id " + std::to_string(id));
    m_StartEvent = std::make_unique<CudaEvent>();
    m_EndEvent = std::make_unique<CudaEvent>();
}

void CudaTransferAction::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void CudaTransferAction::WaitForFinish()
{
    Logger::LogDebug("Waiting for CUDA transfer action with id " + std::to_string(m_Id));
    m_EndEvent->WaitForFinish();
}

TransferActionId CudaTransferAction::GetId() const
{
    return m_Id;
}

QueueId CudaTransferAction::GetQueueId() const
{
    return m_QueueId;
}

CUevent CudaTransferAction::GetStartEvent() const
{
    return m_StartEvent->GetEvent();
}

CUevent CudaTransferAction::GetEndEvent() const
{
    return m_EndEvent->GetEvent();
}

Nanoseconds CudaTransferAction::GetDuration() const
{
    return CudaEvent::GetDuration(*m_StartEvent, *m_EndEvent);
}

Nanoseconds CudaTransferAction::GetOverhead() const
{
    return m_Overhead;
}

TransferResult CudaTransferAction::GenerateResult() const
{
    const Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();
    return TransferResult(duration, overhead);
}

} // namespace ktt

#endif // KTT_API_CUDA
