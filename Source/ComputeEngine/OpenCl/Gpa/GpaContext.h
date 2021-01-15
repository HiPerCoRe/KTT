#pragma once

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <string>
#include <vector>
#include <GPUPerfAPI.h>

#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>

namespace ktt
{

class GpaContext
{
public:
    explicit GpaContext(GPAFunctionTable& functions, OpenClCommandQueue& queue);
    ~GpaContext();

    gpa_uint32 GenerateSampleId();
    void SetCounters(const std::vector<std::string>& counters);

    GPA_ContextId GetContext() const;
    const std::vector<std::string>& GetCounters() const;

private:
    GPAFunctionTable& m_Functions;
    OpenClCommandQueue& m_Queue;
    GPA_ContextId m_Context;
    std::vector<std::string> m_Counters;
    gpa_uint32 m_SampleIdGenerator;

    static const std::vector<std::string>& GetDefaultCounters();
};

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
