#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <optional>
#include <string>

#include <Api/Output/ComputationResult.h>
#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClComputeAction
{
public:
    OpenClComputeAction(const ComputeActionId id, const QueueId queueId, std::shared_ptr<OpenClKernel> kernel,
        const DimensionVector& globalSize, const DimensionVector& localSize);

    void IncreaseOverhead(const Nanoseconds overhead);
    void IncreaseCompilationOverhead(const Nanoseconds overhead);
    void IncreaseProfilingOverhead(const Nanoseconds overhead);
    void IncreasePreciseMeasurementOverhead(const Nanoseconds overhead);
    void SetComputeId(const KernelComputeId& id);
    void SetReleaseFlag();
    void SetDurationFromMultirun(const Nanoseconds duration);
    void SetDurationStdev(const double durationStdev);
    void WaitForFinish();

    ComputeActionId GetId() const;
    QueueId GetQueueId() const;
    OpenClKernel& GetKernel();
    cl_event* GetEvent();
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    Nanoseconds GetCompilationOverhead() const;
    Nanoseconds GetProfilingOverhead() const;
    Nanoseconds GetPreciseMeasurementOverhead() const;
    const KernelComputeId& GetComputeId() const;
    ComputationResult GenerateResult() const;

private:
    ComputeActionId m_Id;
    QueueId m_QueueId;
    std::shared_ptr<OpenClKernel> m_Kernel;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Overhead;
    Nanoseconds m_CompilationOverhead;
    Nanoseconds m_ProfilingOverhead;
    Nanoseconds m_PreciseMeasurementOverhead;
    KernelComputeId m_ComputeId;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    std::optional<Nanoseconds> m_DurationFromMultirun;
    std::optional<double> m_DurationStdev;
};

} // namespace ktt

#endif // KTT_API_OPENCL
