#pragma once

#include <limits>
#include <cupti_profiler_target.h>
#include <compute_engine/cuda/cupti/cupti_metric_configuration.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUPTIProfilingInstance
{
public:
    explicit CUPTIProfilingInstance(CUcontext context, const CUPTIMetricConfiguration& configuration) :
        context(context),
        kernelDuration(std::numeric_limits<uint64_t>::max()),
        configuration(configuration)
    {
        CUpti_Profiler_BeginSession_Params beginParams =
        {
            CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
            nullptr,
            context,
            configuration.counterDataImage.size(),
            this->configuration.counterDataImage.data(),
            configuration.scratchBuffer.size(),
            this->configuration.scratchBuffer.data(),
            0,
            nullptr,
            CUpti_ProfilerRange::CUPTI_AutoRange,
            CUpti_ProfilerReplayMode::CUPTI_UserReplay,
            configuration.maxProfiledRanges,
            configuration.maxProfiledRanges
        };

        checkCUPTIError(cuptiProfilerBeginSession(&beginParams), "cuptiProfilerBeginSession");

        CUpti_Profiler_SetConfig_Params setParams =
        {
            CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
            nullptr,
            context,
            this->configuration.configImage.data(),
            configuration.configImage.size()
        };

        setParams.passIndex = 0;
        checkCUPTIError(cuptiProfilerSetConfig(&setParams), "cuptiProfilerSetConfig");
    }

    ~CUPTIProfilingInstance()
    {
        CUpti_Profiler_UnsetConfig_Params unsetParams =
        {
            CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
            nullptr,
            context
        };

        checkCUPTIError(cuptiProfilerUnsetConfig(&unsetParams), "cuptiProfilerUnsetConfig");

        CUpti_Profiler_EndSession_Params endParams =
        {
            CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
            nullptr,
            context
        };

        checkCUPTIError(cuptiProfilerEndSession(&endParams), "cuptiProfilerEndSession");
    }

    void collectData()
    {
        CUpti_Profiler_FlushCounterData_Params flushParams =
        {
            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
            nullptr,
            context
        };

        checkCUPTIError(cuptiProfilerFlushCounterData(&flushParams), "cuptiProfilerFlushCounterData");
        configuration.dataCollected = true;
    }

    void setKernelDuration(const uint64_t duration)
    {
        kernelDuration = duration;
    }

    bool isDataReady() const
    {
        return configuration.dataCollected;
    }

    uint64_t getRemainingPasses() const
    {
        if (isDataReady())
        {
            return 0;
        }

        // Currently there is no way to retrieve total number of passes needed in new CUPTI API
        return 1;
    }

    CUcontext getContext() const
    {
        return context;
    }

    bool hasValidKernelDuration() const
    {
        return kernelDuration != std::numeric_limits<uint64_t>::max();
    }

    uint64_t getKernelDuration() const
    {
        return kernelDuration;
    }

    const CUPTIMetricConfiguration& getMetricConfiguration() const
    {
        return configuration;
    }

private:
    CUcontext context;
    uint64_t kernelDuration;
    CUPTIMetricConfiguration configuration;
};

} // namespace ktt
