#pragma once

#include <compute_engine/cuda/cupti/cupti_profiling_instance.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUPTIProfilingPass
{
public:
    explicit CUPTIProfilingPass(CUPTIProfilingInstance& instance) :
        instance(&instance)
    {
        CUpti_Profiler_BeginPass_Params beginParams =
        {
            CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
            nullptr,
            instance.getContext()
        };

        checkCUPTIError(cuptiProfilerBeginPass(&beginParams), "cuptiProfilerBeginPass");

        CUpti_Profiler_EnableProfiling_Params enableParams =
        {
            CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
            nullptr,
            instance.getContext()
        };

        checkCUPTIError(cuptiProfilerEnableProfiling(&enableParams), "cuptiProfilerEnableProfiling");
    }

    ~CUPTIProfilingPass()
    {
        CUpti_Profiler_DisableProfiling_Params disableParams =
        {
            CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
            nullptr,
            instance->getContext()
        };

        checkCUPTIError(cuptiProfilerDisableProfiling(&disableParams), "cuptiProfilerDisableProfiling");

        CUpti_Profiler_EndPass_Params endParams =
        {
            CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
            nullptr,
            instance->getContext()
        };

        checkCUPTIError(cuptiProfilerEndPass(&endParams), "cuptiProfilerEndPass");

        if (endParams.allPassesSubmitted)
        {
            instance->collectData();
        }
    }

private:
    CUPTIProfilingInstance* instance;
};

} // namespace ktt
