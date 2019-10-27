#pragma once

#include <stdexcept>
#include <vector>
#include <cupti_profiler_target.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUPTIProfiler
{
public:
    explicit CUPTIProfiler()
    {
        CUpti_Profiler_Initialize_Params params =
        {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
            nullptr
        };

        checkCUPTIError(cuptiProfilerInitialize(&params), "cuptiProfilerInitialize");
    }

    ~CUPTIProfiler()
    {
        CUpti_Profiler_DeInitialize_Params params =
        {
            CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
            nullptr
        };

        checkCUPTIError(cuptiProfilerDeInitialize(&params), "cuptiProfilerDeInitialize");
    }
};

} // namespace ktt
