#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <optional>
#include <string>

#include <Api//Configuration/DimensionVector.h>
#include <Api/Output/ComputationResult.h>
#include <ComputeEngine/Cuda/CudaEvent.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <KttTypes.h>

namespace ktt
{

class CudaComputeAction
{
public:
    CudaComputeAction(const ComputeActionId id, const QueueId queueId, std::shared_ptr<CudaKernel> kernel,
        const DimensionVector& globalSize, const DimensionVector& localSize);

    void IncreaseOverhead(const Nanoseconds overhead);
    void SetComputeId(const KernelComputeId& id);
    void SetPowerUsage(const uint32_t powerUsage);
    void WaitForFinish();

    ComputeActionId GetId() const;
    QueueId GetQueueId() const;
    CudaKernel& GetKernel();
    CUevent GetStartEvent() const;
    CUevent GetEndEvent() const;
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    const KernelComputeId& GetComputeId() const;
    ComputationResult GenerateResult() const;

private:
    ComputeActionId m_Id;
    QueueId m_QueueId;
    std::shared_ptr<CudaKernel> m_Kernel;
    std::unique_ptr<CudaEvent> m_StartEvent;
    std::unique_ptr<CudaEvent> m_EndEvent;
    Nanoseconds m_Overhead;
    KernelComputeId m_ComputeId;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    std::optional<uint32_t> m_PowerUsage;
};

} // namespace ktt

#endif // KTT_API_CUDA
