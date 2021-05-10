#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/Gpa/GpaInterface.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/Logger/Logger.h>

GPAApiManager* GPAApiManager::m_pGpaApiManager = nullptr;
GPAFuncTableInfo* g_pFuncTableInfo = nullptr;

namespace ktt
{

GpaInterface::GpaInterface()
{
    Logger::LogDebug("Initializing GPA API");

    if (GPAApiManager::Instance()->LoadApi(GPA_API_OPENCL) != GPA_STATUS_OK)
    {
        throw KttException("GPA profiling API error: Failed to load GPA .dll (.so) library");
    }

    m_Functions = GPAApiManager::Instance()->GetFunctionTable(GPA_API_OPENCL);

    if (m_Functions == nullptr)
    {
        throw KttException("GPA profiling API error: Failed to retrieve GPA function table");
    }

    CheckError(m_Functions->GPA_Initialize(GPA_INITIALIZE_DEFAULT_BIT), *m_Functions, "GPA_Initialize");

#if defined(KTT_CONFIGURATION_DEBUG)
    CheckError(m_Functions->GPA_RegisterLoggingCallback(GPA_LOGGING_ERROR_MESSAGE_AND_TRACE, LoggingCallback), *m_Functions,
        "GPA_RegisterLoggingCallback");
#endif
}

GpaInterface::~GpaInterface()
{
    Logger::LogDebug("Releasing GPA API");
    GPAApiManager::Instance()->UnloadApi(GPA_API_OPENCL);
}

GPAFunctionTable& GpaInterface::GetFunctions()
{
    return *m_Functions;
}

void GpaInterface::LoggingCallback(GPA_Logging_Type messageType, const char* message)
{
    switch (messageType)
    {
    case GPA_LOGGING_ERROR:
        Logger::LogError(message);
        break;
    case GPA_LOGGING_DEBUG_MESSAGE:
        Logger::LogInfo(message);
        break;
    case GPA_LOGGING_DEBUG_TRACE:
        Logger::LogDebug(message);
        break;
    default:
        Logger::LogDebug(message);
    }
}

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
