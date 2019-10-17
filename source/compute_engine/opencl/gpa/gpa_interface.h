#pragma once

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#ifdef KTT_PROFILING_GPA
#include <gpu_perf_api/GPAInterfaceLoader.h>
#include <gpu_perf_api/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA

#ifdef KTT_PROFILING_GPA_LEGACY
#include <gpu_perf_api_legacy/GPAInterfaceLoader.h>
#include <gpu_perf_api_legacy/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA_LEGACY

namespace ktt
{

class GPAInterface
{
public:
    GPAInterface();
    ~GPAInterface();

    GPAFunctionTable& getFunctionTable();

private:
    GPAFunctionTable* gpaFunctionTable;

    static void gpaLoggingCallback(GPA_Logging_Type messageType, const char* message);
};

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
