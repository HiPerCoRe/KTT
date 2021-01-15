#pragma once

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <GPAInterfaceLoader.h>
#include <GPUPerfAPI.h>

namespace ktt
{

class GpaInterface
{
public:
    GpaInterface();
    ~GpaInterface();

    GPAFunctionTable& GetFunctions();

private:
    GPAFunctionTable* m_Functions;

    static void LoggingCallback(GPA_Logging_Type messageType, const char* message);
};

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
