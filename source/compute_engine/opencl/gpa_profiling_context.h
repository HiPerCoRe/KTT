#pragma once

#include <vector>
#include <gpu_perf_api/GPUPerfAPI.h>
#include <compute_engine/opencl/opencl_command_queue.h>
#include <compute_engine/opencl/opencl_utility.h>

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
