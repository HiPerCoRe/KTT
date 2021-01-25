#ifdef KTT_PROFILING_CUPTI

#include <ComputeEngine/Cuda/Cupti/CuptiMetricConfiguration.h>

namespace ktt
{

CuptiMetricConfiguration::CuptiMetricConfiguration(const uint32_t maxProfiledRanges) :
    m_MaxProfiledRanges(maxProfiledRanges),
    m_DataCollected(false)
{}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
