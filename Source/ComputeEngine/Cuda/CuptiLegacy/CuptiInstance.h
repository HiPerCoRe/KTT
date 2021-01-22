#pragma once

#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <memory>
#include <vector>

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiMetric.h>
#include <KttTypes.h>

namespace ktt
{

class CudaContext;
class KernelProfilingData;

class CuptiInstance
{
public:
    explicit CuptiInstance(const CudaContext& context);
    ~CuptiInstance();

    void UpdatePassIndex();
    void SetKernelDuration(const Nanoseconds duration);

    const CudaContext& GetContext() const;
    Nanoseconds GetKernelDuration() const;
    bool HasValidKernelDuration() const;
    uint64_t GetRemainingPassCount() const;
    uint64_t GetTotalPassCount() const;
    uint32_t GetCurrentIndex() const;
    CUpti_EventGroupSets& GetEventSets();
    std::vector<std::unique_ptr<CuptiMetric>>& GetMetrics();
    std::unique_ptr<KernelProfilingData> GenerateProfilingData();

    static void SetEnabledMetrics(const std::vector<std::string>& metrics);

private:
    const CudaContext& m_Context;
    Nanoseconds m_KernelDuration;
    uint64_t m_CurrentPassIndex;
    uint64_t m_TotalPassCount;
    CUpti_EventGroupSets* m_EventSets;
    std::vector<std::unique_ptr<CuptiMetric>> m_Metrics;

    inline static std::vector<std::string> m_EnabledMetrics;

    static const std::vector<std::string>& GetDefaultMetrics();
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
