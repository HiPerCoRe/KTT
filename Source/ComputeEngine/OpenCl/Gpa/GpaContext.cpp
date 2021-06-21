#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/Gpa/GpaContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

GpaContext::GpaContext(GPAFunctionTable& functions, OpenClCommandQueue& queue) :
    m_Functions(functions),
    m_Queue(queue),
    m_SampleIdGenerator(0)
{
    Logger::LogDebug("Initializing GPA context");
    CheckError(functions.GPA_OpenContext(queue.GetQueue(), GPA_OPENCONTEXT_DEFAULT_BIT, &m_Context), functions,
        "GPA_OpenContext");
    SetCounters(GetDefaultCounters());
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

    std::vector<std::string> filteredCounters;

    for (const auto& counter : counters)
    {
        [[maybe_unused]] gpa_uint32 index;
        const GPA_Status status = m_Functions.GPA_GetCounterIndex(m_Context, counter.c_str(), &index);

        if (status != GPA_STATUS_OK)
        {
            Logger::LogWarning("Profiling counter with name " + counter + " is not supported on the current device");
            continue;
        }

        filteredCounters.push_back(counter);
    }

    if (filteredCounters.empty())
    {
        throw KttException("No valid counters provided for the current device");
    }

    m_Counters = filteredCounters;
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
