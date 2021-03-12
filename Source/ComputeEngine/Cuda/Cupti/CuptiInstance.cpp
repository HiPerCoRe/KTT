#ifdef KTT_PROFILING_CUPTI

#include <cupti_profiler_target.h>

#include <ComputeEngine/Cuda/Cupti/CuptiInstance.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiInstance::CuptiInstance(const CudaContext& context, const CuptiMetricConfiguration& configuration) :
    m_Context(context),
    m_Configuration(configuration)
{
    Logger::LogDebug("Initializing CUPTI instance");

    CUpti_Profiler_BeginSession_Params beginParams =
    {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        nullptr,
        m_Context.GetContext(),
        m_Configuration.m_CounterDataImage.size(),
        m_Configuration.m_CounterDataImage.data(),
        m_Configuration.m_ScratchBuffer.size(),
        m_Configuration.m_ScratchBuffer.data(),
        0,
        nullptr,
        CUpti_ProfilerRange::CUPTI_AutoRange,
        CUpti_ProfilerReplayMode::CUPTI_UserReplay,
        m_Configuration.m_MaxProfiledRanges,
        m_Configuration.m_MaxProfiledRanges
    };

    CheckError(cuptiProfilerBeginSession(&beginParams), "cuptiProfilerBeginSession");

    CUpti_Profiler_SetConfig_Params setParams =
    {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
        nullptr,
        m_Context.GetContext(),
        m_Configuration.m_ConfigImage.data(),
        m_Configuration.m_ConfigImage.size(),
        1,
        1,
        0,
        1
    };

    CheckError(cuptiProfilerSetConfig(&setParams), "cuptiProfilerSetConfig");
}

CuptiInstance::~CuptiInstance()
{
    Logger::LogDebug("Releasing CUPTI instance");

    CUpti_Profiler_UnsetConfig_Params unsetParams =
    {
        CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
        nullptr,
        m_Context.GetContext()
    };

    CheckError(cuptiProfilerUnsetConfig(&unsetParams), "cuptiProfilerUnsetConfig");

    CUpti_Profiler_EndSession_Params endParams =
    {
        CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
        nullptr,
        m_Context.GetContext()
    };

    CheckError(cuptiProfilerEndSession(&endParams), "cuptiProfilerEndSession");
}

void CuptiInstance::CollectData()
{
    CUpti_Profiler_FlushCounterData_Params flushParams =
    {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
        nullptr,
        m_Context.GetContext(),
        0,
        0
    };

    CheckError(cuptiProfilerFlushCounterData(&flushParams), "cuptiProfilerFlushCounterData");
    m_Configuration.m_DataCollected = true;
}

const CudaContext& CuptiInstance::GetContext() const
{
    return m_Context;
}

uint64_t CuptiInstance::GetRemainingPassCount() const
{
    if (IsDataReady())
    {
        return 0;
    }

    // Currently, there is no way to retrieve total number of passes needed in new CUPTI API
    return 1;
}

bool CuptiInstance::IsDataReady() const
{
    return m_Configuration.m_DataCollected;
}

const CuptiMetricConfiguration& CuptiInstance::GetConfiguration() const
{
    return m_Configuration;
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
