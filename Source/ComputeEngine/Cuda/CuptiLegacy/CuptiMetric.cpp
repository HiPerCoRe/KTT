#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <algorithm>

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiMetric.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

CuptiMetric::CuptiMetric(const CUpti_MetricID id, const std::string& name) :
    m_Id(id),
    m_Name(name)
{
    uint32_t eventCount;
    CheckError(cuptiMetricGetNumEvents(m_Id, &eventCount), "cuptiMetricGetNumEvents");
    m_EventIds.resize(static_cast<size_t>(eventCount));
    m_EventValues.resize(static_cast<size_t>(eventCount));
    m_EventStates.resize(static_cast<size_t>(eventCount), false);

    size_t eventsSize = sizeof(CUpti_EventID) * m_EventIds.size();
    CheckError(cuptiMetricEnumEvents(m_Id, &eventsSize, m_EventIds.data()), "cuptiMetricEnumEvents");
}

void CuptiMetric::SetEventValue(const CUpti_EventID id, const uint64_t value)
{
    for (size_t i = 0; i < m_EventIds.size(); ++i)
    {
        if (m_EventIds[i] == id)
        {
            m_EventValues[i] = value;
            m_EventStates[i] = true;
            break;
        }
    }
}

CUpti_MetricID CuptiMetric::GetId() const
{
    return m_Id;
}

const std::string& CuptiMetric::GetName() const
{
    return m_Name;
}

KernelProfilingCounter CuptiMetric::GenerateCounter(const CudaContext& context, const Nanoseconds kernelDuration)
{
    [[maybe_unused]] const bool allStatesValid = std::all_of(m_EventStates.cbegin(), m_EventStates.cend(), [](const bool state)
    {
        return state;
    });

    KttAssert(allStatesValid, "Some metric events for profiling counter calculation were not collected");

    CUpti_MetricValue metricValue;
    CheckError(cuptiMetricGetValue(context.GetDevice(), m_Id, m_EventIds.size() * sizeof(CUpti_EventID), m_EventIds.data(),
        m_EventValues.size() * sizeof(uint64_t), m_EventValues.data(), kernelDuration, &metricValue), "cuptiMetricGetValue");

    CUpti_MetricValueKind valueKind;
    size_t valueKindSize = sizeof(valueKind);
    CheckError(cuptiMetricGetAttribute(m_Id, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind),
        "cuptiMetricGetAttribute");

    switch (valueKind)
    {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::Double, metricValue.metricValueDouble);
    case CUPTI_METRIC_VALUE_KIND_UINT64:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::UnsignedInt, metricValue.metricValueUint64);
    case CUPTI_METRIC_VALUE_KIND_INT64:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::Int, metricValue.metricValueInt64);
    case CUPTI_METRIC_VALUE_KIND_PERCENT:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::Percent, metricValue.metricValuePercent);
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::Throughput, metricValue.metricValueThroughput);
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        return KernelProfilingCounter(m_Name, ProfilingCounterType::UtilizationLevel,
            static_cast<uint64_t>(metricValue.metricValueUtilizationLevel));
    default:
        KttError("Unhandled CUPTI metric type");
        return KernelProfilingCounter("", ProfilingCounterType::UnsignedInt, static_cast<uint64_t>(0));
    }
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
