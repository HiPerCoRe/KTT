#ifdef KTT_API_CUDA

#include <string>

#include <ComputeEngine/Cuda/Actions/CudaComputeAction.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaComputeAction::CudaComputeAction(const ComputeActionId id, const QueueId queueId, std::shared_ptr<CudaKernel> kernel,
    const DimensionVector& globalSize, const DimensionVector& localSize) :
    m_Id(id),
    m_QueueId(queueId),
    m_Kernel(kernel),
    m_Overhead(0),
    m_CompilationOverhead(0),
    m_GlobalSize(globalSize),
    m_LocalSize(localSize)
{
    Logger::LogDebug("Initializing CUDA compute action with id " + std::to_string(id)
        + " for kernel with name " + kernel->GetName());
    KttAssert(m_Kernel != nullptr, "Invalid kernel was used during CUDA compute action initialization");

    m_StartEvent = std::make_unique<CudaEvent>();
    m_EndEvent = std::make_unique<CudaEvent>();
}

void CudaComputeAction::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void CudaComputeAction::IncreaseCompilationOverhead(const Nanoseconds overhead)
{
    m_CompilationOverhead += overhead;
}


void CudaComputeAction::SetComputeId(const KernelComputeId& id)
{
    m_ComputeId = id;
}

void CudaComputeAction::SetPowerUsage(const uint32_t powerUsage)
{
    m_PowerUsage = powerUsage;
}

void CudaComputeAction::WaitForFinish()
{
    Logger::LogDebug("Waiting for CUDA kernel compute action with id " + std::to_string(m_Id));
    m_EndEvent->WaitForFinish();
}

ComputeActionId CudaComputeAction::GetId() const
{
    return m_Id;
}

QueueId CudaComputeAction::GetQueueId() const
{
    return m_QueueId;
}

CudaKernel& CudaComputeAction::GetKernel()
{
    return *m_Kernel;
}

CUevent CudaComputeAction::GetStartEvent() const
{
    return m_StartEvent->GetEvent();
}

CUevent CudaComputeAction::GetEndEvent() const
{
    return m_EndEvent->GetEvent();
}

Nanoseconds CudaComputeAction::GetDuration() const
{
    return CudaEvent::GetDuration(*m_StartEvent, *m_EndEvent);
}

Nanoseconds CudaComputeAction::GetOverhead() const
{
    return m_Overhead;
}

Nanoseconds CudaComputeAction::GetCompilationOverhead() const
{
    return m_CompilationOverhead;
}

const KernelComputeId& CudaComputeAction::GetComputeId() const
{
    return m_ComputeId;
}

ComputationResult CudaComputeAction::GenerateResult() const
{
    ComputationResult result(m_Kernel->GetName());
    const Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();
    const Nanoseconds compilationOverhead = GetCompilationOverhead();
    std::unique_ptr<KernelCompilationData> compilationData = m_Kernel->GenerateCompilationData();

    result.SetDurationData(duration, overhead, compilationOverhead);
    result.SetSizeData(m_GlobalSize, m_LocalSize);
    result.SetCompilationData(std::move(compilationData));
    
    if (m_PowerUsage.has_value())
    {
        result.SetPowerUsage(m_PowerUsage.value());
    }

    return result;
}

} // namespace ktt

#endif // KTT_API_CUDA
