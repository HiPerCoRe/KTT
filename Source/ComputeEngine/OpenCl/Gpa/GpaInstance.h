#pragma once

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <memory>
#include <GPUPerfAPI.h>

#include <Api/Output/KernelProfilingCounter.h>
#include <ComputeEngine/OpenCl/Gpa/GpaContext.h>

namespace ktt
{

class KernelProfilingData;

class GpaInstance
{
public:
    explicit GpaInstance(GPAFunctionTable& functions, GpaContext& context);
    ~GpaInstance();

    void UpdatePassIndex();

    GPA_SessionId GetSession() const;
    gpa_uint32 GetSampleId() const;
    gpa_uint32 GetPassIndex() const;
    gpa_uint32 GetRemainingPassCount() const;
    std::unique_ptr<KernelProfilingData> GenerateProfilingData() const;

private:
    GPAFunctionTable& m_Functions;
    GpaContext& m_Context;
    GPA_SessionId m_Session;
    gpa_uint32 m_SampleId;
    gpa_uint32 m_CurrentPassIndex;
    gpa_uint32 m_TotalPassCount;

    KernelProfilingCounter GenerateCounterForIndex(const gpa_uint32 index, uint64_t sampleData) const;
};

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
