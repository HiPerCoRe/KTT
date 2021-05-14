#if defined(KTT_PROFILING_CUPTI)

#include <cupti_profiler_target.h>

#include <ComputeEngine/Cuda/Cupti/CuptiProfiler.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiProfiler::CuptiProfiler()
{
    Logger::LogDebug("Initializing CUPTI profiler");

    CUpti_Profiler_Initialize_Params params =
    {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(cuptiProfilerInitialize(&params), "cuptiProfilerInitialize");
}

CuptiProfiler::~CuptiProfiler()
{
    Logger::LogDebug("Releasing CUPTI profiler");

    CUpti_Profiler_DeInitialize_Params params =
    {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(cuptiProfilerDeInitialize(&params), "cuptiProfilerDeInitialize");
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
