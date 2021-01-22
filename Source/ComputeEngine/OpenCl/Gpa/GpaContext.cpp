#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <ComputeEngine/OpenCl/Gpa/GpaContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

GpaContext::GpaContext(GPAFunctionTable& functions, OpenClCommandQueue& queue) :
    m_Functions(functions),
    m_Queue(queue),
    m_Counters(GetDefaultCounters()),
    m_SampleIdGenerator(0)
{
    Logger::LogDebug("Initializing GPA context");
    CheckError(functions.GPA_OpenContext(queue.GetQueue(), GPA_OPENCONTEXT_DEFAULT_BIT, &m_Context), functions,
        "GPA_OpenContext");
}

GpaContext::~GpaContext()
{
    Logger::LogDebug("Releasing GPA context");
    CheckError(m_Functions.GPA_CloseContext(m_Context), m_Functions, "GPA_CloseContext");
}

gpa_uint32 GpaContext::GenerateSampleId()
{
    return m_SampleIdGenerator++;
}

void GpaContext::SetCounters(const std::vector<std::string>& counters)
{
    if (counters.empty())
    {
        throw KttException("Number of profiling counters must be greater than zero");
    }

    m_Counters = counters;
}

GPA_ContextId GpaContext::GetContext() const
{
    return m_Context;
}

const std::vector<std::string>& GpaContext::GetCounters() const
{
    return m_Counters;
}

const std::vector<std::string>& GpaContext::GetDefaultCounters()
{
    static const std::vector<std::string> result
    {
        "Wavefronts",
        "VALUInsts",
        "SALUInsts",
        "VFetchInsts",
        "SFetchInsts",
        "VWriteInsts",
        "VALUUtilization",
        "VALUBusy",
        "SALUBusy",
        "FetchSize",
        "WriteSize",
        "MemUnitBusy",
        "MemUnitStalled",
        "WriteUnitStalled"
    };

    return result;
}

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
