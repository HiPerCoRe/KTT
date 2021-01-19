#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <string>

#include <Api/Output/KernelProfilingData.h>
#include <ComputeEngine/OpenCl/Gpa/GpaInstance.h>
#include <ComputeEngine/OpenCl/Gpa/GpaPass.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

GpaInstance::GpaInstance(GPAFunctionTable& functions, GpaContext& context) :
    m_Functions(functions),
    m_Context(context),
    m_SampleId(context.GenerateSampleId()),
    m_CurrentPassIndex(0)
{
    Logger::LogDebug("Initializing GPA instance with sample id " + std::to_string(m_SampleId));
    CheckError(functions.GPA_CreateSession(context.GetContext(), GPA_SESSION_SAMPLE_TYPE_DISCRETE_COUNTER, &m_Session),
        functions, "GPA_CreateSession");

    for (const auto& counter : context.GetCounters())
    {
        CheckError(functions.GPA_EnableCounterByName(m_Session, counter.c_str()), functions, "GPA_EnableCounterByName");
    }

    CheckError(functions.GPA_BeginSession(m_Session), functions, "GPA_BeginSession");
    CheckError(functions.GPA_GetPassCount(m_Session, &m_TotalPassCount), functions, "GPA_GetPassCount");
}

GpaInstance::~GpaInstance()
{
    Logger::LogDebug("Releasing GPA instance with sample id " + std::to_string(m_SampleId));

    // There is currently no way to abort active profiling session without performing all passes, so dummy passes are launched
    if (GetRemainingPassCount() > 0)
    {
        while (GetRemainingPassCount() > 0)
        {
            [[maybe_unused]] auto pass = std::make_unique<GpaPass>(m_Functions, *this);
        }

        CheckError(m_Functions.GPA_EndSession(m_Session), m_Functions, "GPA_EndSession");
    }

    CheckError(m_Functions.GPA_DeleteSession(m_Session), m_Functions, "GPA_DeleteSession");
}

void GpaInstance::UpdatePassIndex()
{
    ++m_CurrentPassIndex;
}

GPA_SessionId GpaInstance::GetSession() const
{
    return m_Session;
}

gpa_uint32 GpaInstance::GetSampleId() const
{
    return m_SampleId;
}

gpa_uint32 GpaInstance::GetPassIndex() const
{
    return m_CurrentPassIndex;
}

gpa_uint32 GpaInstance::GetRemainingPassCount() const
{
    if (m_CurrentPassIndex > m_TotalPassCount)
    {
        return 0;
    }

    return m_TotalPassCount - m_CurrentPassIndex;
}

std::unique_ptr<KernelProfilingData> GpaInstance::GenerateProfilingData() const
{
    const auto remainingCount = static_cast<uint64_t>(GetRemainingPassCount());

    if (remainingCount > 0)
    {
        return std::make_unique<KernelProfilingData>(remainingCount);
    }

    Logger::LogDebug("Generating profiling data for GPA instance with sample id " + std::to_string(m_SampleId));

    CheckError(m_Functions.GPA_EndSession(m_Session), m_Functions, "GPA_EndSession");
    KttAssert(m_Functions.GPA_IsSessionComplete(m_Session) == GPA_STATUS_OK, "Incorrect handling of GPA profiling session");

    size_t sampleSize;
    CheckError(m_Functions.GPA_GetSampleResultSize(m_Session, m_SampleId, &sampleSize), m_Functions, "GPA_GetSampleResultSize");

    std::vector<uint64_t> sampleData(sampleSize / sizeof(uint64_t));
    CheckError(m_Functions.GPA_GetSampleResult(m_Session, m_SampleId, sampleSize, sampleData.data()), m_Functions,
        "GPA_GetSampleResult");

    gpa_uint32 enabledCount;
    CheckError(m_Functions.GPA_GetNumEnabledCounters(m_Session, &enabledCount), m_Functions, "GPA_GetNumEnabledCounters");

    std::vector<KernelProfilingCounter> counters;

    for (gpa_uint32 i = 0; i < enabledCount; ++i)
    {
        const KernelProfilingCounter counter = GenerateCounterForIndex(i, sampleData[i]);
        counters.push_back(counter);
    }

    return std::make_unique<KernelProfilingData>(counters);
}

KernelProfilingCounter GpaInstance::GenerateCounterForIndex(const gpa_uint32 index, uint64_t sampleData) const
{
    gpa_uint32 counterIndex;
    CheckError(m_Functions.GPA_GetEnabledIndex(m_Session, index, &counterIndex), m_Functions, "GPA_GetEnabledIndex");

    GPA_Data_Type counterType;
    CheckError(m_Functions.GPA_GetCounterDataType(m_Context.GetContext(), counterIndex, &counterType), m_Functions,
        "GPA_GetCounterDataType");

    const char* counterName;
    CheckError(m_Functions.GPA_GetCounterName(m_Context.GetContext(), counterIndex, &counterName), m_Functions,
        "GPA_GetCounterName");

    switch (counterType)
    {
    case GPA_DATA_TYPE_FLOAT64:
        return KernelProfilingCounter(counterName, ProfilingCounterType::Double, *(reinterpret_cast<double*>(&sampleData)));
    case GPA_DATA_TYPE_UINT64:
        return KernelProfilingCounter(counterName, ProfilingCounterType::UnsignedInt, sampleData);
    default:
        KttError("Unhandled GPA counter type");
        return KernelProfilingCounter("", ProfilingCounterType::UnsignedInt, 0ull);
    }
}

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
