#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstdint>

#include <ComputeEngine/Cuda/Cupti/CuptiMetricConfiguration.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <KttTypes.h>

namespace ktt
{

class CuptiInstance
{
public:
    explicit CuptiInstance(const CudaContext& context, const CuptiMetricConfiguration& configuration);
    ~CuptiInstance();

    void CollectData();
    void SetKernelDuration(const Nanoseconds duration);

    const CudaContext& GetContext() const;
    Nanoseconds GetKernelDuration() const;
    bool HasValidKernelDuration() const;
    uint64_t GetRemainingPassCount() const;
    bool IsDataReady() const;
    const CuptiMetricConfiguration& GetConfiguration() const;

private:
    const CudaContext& m_Context;
    CuptiMetricConfiguration m_Configuration;
    Nanoseconds m_KernelDuration;
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
