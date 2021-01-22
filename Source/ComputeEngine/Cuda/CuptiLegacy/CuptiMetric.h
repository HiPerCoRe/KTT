#pragma once

#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <cstdint>
#include <string>
#include <vector>

#include <Api/Output/KernelProfilingCounter.h>
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>
#include <KttTypes.h>

namespace ktt
{

class CudaContext;

class CuptiMetric
{
public:
    explicit CuptiMetric(const CUpti_MetricID id, const std::string& name);

    void SetEventValue(const CUpti_EventID id, const uint64_t value);

    CUpti_MetricID GetId() const;
    const std::string& GetName() const;
    KernelProfilingCounter GenerateCounter(const CudaContext& context, Nanoseconds kernelDuration);

private:
    CUpti_MetricID m_Id;
    std::string m_Name;
    std::vector<CUpti_EventID> m_EventIds;
    std::vector<uint64_t> m_EventValues;
    std::vector<bool> m_EventStates;
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
