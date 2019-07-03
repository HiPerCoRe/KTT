#pragma once

#include <gpu_perf_api/GPUPerfAPI.h>
#include <compute_engine/opencl/gpa_profiling_instance.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

class GPAProfilingPass
{
public:
    explicit GPAProfilingPass(GPAFunctionTable& gpaFunctions, GPAProfilingInstance& instance) :
        gpaFunctions(gpaFunctions),
        instance(instance),
        passIndex(instance.getCurrentPassIndex())
    {
        checkGPAError(gpaFunctions.GPA_BeginCommandList(instance.getSession(), passIndex, GPA_NULL_COMMAND_LIST, GPA_COMMAND_LIST_NONE, &commandList),
            "GPA_BeginCommandList");
        checkGPAError(gpaFunctions.GPA_BeginSample(instance.getSampleId(), commandList), "GPA_BeginSample");
    }

    ~GPAProfilingPass()
    {
        checkGPAError(gpaFunctions.GPA_EndSample(commandList), "GPA_EndSample");
        checkGPAError(gpaFunctions.GPA_EndCommandList(commandList), "GPA_EndCommandList");
    }

private:
    GPAFunctionTable& gpaFunctions;
    GPAProfilingInstance& instance;
    gpa_uint32 passIndex;
    GPA_CommandListId commandList;
};

} // namespace ktt
