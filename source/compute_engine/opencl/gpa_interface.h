#pragma once

#ifdef KTT_PROFILING_AMD

#include <gpu_perf_api/GPAInterfaceLoader.h>
#include <gpu_perf_api/GPUPerfAPI.h>

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

#endif // KTT_PROFILING_AMD
