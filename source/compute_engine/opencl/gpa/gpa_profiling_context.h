#pragma once

#include <vector>
#include <compute_engine/opencl/opencl_command_queue.h>
#include <compute_engine/opencl/opencl_utility.h>

#ifdef KTT_PROFILING_GPA
#include <gpu_perf_api/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA

#ifdef KTT_PROFILING_GPA_LEGACY
#include <gpu_perf_api_legacy/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA_LEGACY

namespace ktt
{

class GPAProfilingContext
{
public:
    explicit GPAProfilingContext(GPAFunctionTable& gpaFunctions, OpenCLCommandQueue& queue) :
        gpaFunctions(gpaFunctions),
        queue(queue),
        currentSampleId(0)
    {
        checkGPAError(gpaFunctions.GPA_OpenContext(queue.getQueue(), GPA_OPENCONTEXT_DEFAULT_BIT, &context), "GPA_OpenContext");
    }

    ~GPAProfilingContext()
    {
        checkGPAError(gpaFunctions.GPA_CloseContext(context), "GPA_CloseContext");
    }

    gpa_uint32 generateNewSampleId()
    {
        return currentSampleId++;
    }

    void setCounters(const std::vector<std::string>& counters)
    {
        this->counters = counters;
    }

    GPA_ContextId getContext()
    {
        return context;
    }

    const std::vector<std::string>& getCounters() const
    {
        return counters;
    }

private:
    GPAFunctionTable& gpaFunctions;
    OpenCLCommandQueue& queue;
    GPA_ContextId context;
    std::vector<std::string> counters;
    gpa_uint32 currentSampleId;
};

} // namespace ktt
