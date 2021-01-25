#ifdef KTT_PROFILING_CUPTI

#include <ComputeEngine/Cuda/Cupti/CuptiMetric.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiMetric::CuptiMetric(const std::string& name) :
    m_Name(name)
{}

void CuptiMetric::SetRangeValue(const std::string& range, const double value)
{
    m_RangeToValue[range] = value;
}

KernelProfilingCounter CuptiMetric::GenerateCounter() const
{
    KttAssert(!m_RangeToValue.empty(), "Attempting to generate profiling counter from uninitialized metric");
    return KernelProfilingCounter(m_Name, ProfilingCounterType::Double, m_RangeToValue.cbegin()->second);
}

void CuptiMetric::PrintData() const
{
    Logger::LogInfo("Metric name: " + m_Name);

    for (const auto& range : m_RangeToValue)
    {
        Logger::LogInfo("GPU value for range " + range.first + ":" + std::to_string(range.second));
    }
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
