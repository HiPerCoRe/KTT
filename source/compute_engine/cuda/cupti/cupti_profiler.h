#pragma once

#include <stdexcept>
#include <vector>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <compute_engine/cuda/cuda_utility.h>
#include <ktt_types.h>

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

    std::string getDeviceName(const DeviceIndex index) const
    {
        CUpti_Device_GetChipName_Params params =
        {
            CUpti_Device_GetChipName_Params_STRUCT_SIZE,
            nullptr,
            static_cast<size_t>(index)
        };

        checkCUPTIError(cuptiDeviceGetChipName(&params), "cuptiDeviceGetChipName");
        return std::string(params.pChipName);
    }
};

} // namespace ktt
