#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <stdexcept>
#include <compute_engine/opencl/gpa/gpa_interface.h>
#include <compute_engine/opencl/opencl_utility.h>
#include <utility/logger.h>

GPAApiManager* GPAApiManager::m_pGpaApiManager = nullptr;
GPAFuncTableInfo* g_pFuncTableInfo = nullptr;

namespace ktt
{

GPAInterface::GPAInterface()
{
    checkGPAError(GPAApiManager::Instance()->LoadApi(GPA_API_OPENCL), "LoadApi");
    gpaFunctionTable = GPAApiManager::Instance()->GetFunctionTable(GPA_API_OPENCL);

    if (gpaFunctionTable == nullptr)
    {
        throw std::runtime_error("GPA profiling API error: Failed to retrieve GPA function table");
    }

    checkGPAError(gpaFunctionTable->GPA_Initialize(GPA_INITIALIZE_DEFAULT_BIT), "GPA_Initialize");
    checkGPAError(gpaFunctionTable->GPA_RegisterLoggingCallback(GPA_LOGGING_ERROR_MESSAGE_AND_TRACE, gpaLoggingCallback),
        "GPA_RegisterLoggingCallback");
}

GPAInterface::~GPAInterface()
{
    GPAApiManager::Instance()->UnloadApi(GPA_API_OPENCL);
}

GPAFunctionTable& GPAInterface::getFunctionTable()
{
    return *gpaFunctionTable;
}

void GPAInterface::gpaLoggingCallback(GPA_Logging_Type messageType, const char* message)
{
    switch (messageType)
    {
    case GPA_LOGGING_ERROR:
        Logger::logError(message);
        break;
    case GPA_LOGGING_DEBUG_MESSAGE:
        Logger::logInfo(message);
        break;
    case GPA_LOGGING_DEBUG_TRACE:
        Logger::logDebug(message);
        break;
    default:
        Logger::logDebug(message);
    }
}

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
