#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstddef>
#include <string>
#include <vector>

namespace ktt
{

struct CuptiMetricConfiguration
{
public:
    CuptiMetricConfiguration(const uint32_t maxProfiledRanges);

    std::vector<std::string> m_MetricNames;
    std::vector<uint8_t> m_ConfigImage;
    std::vector<uint8_t> m_ScratchBuffer;
    std::vector<uint8_t> m_CounterDataImage;
    uint32_t m_MaxProfiledRanges;
    bool m_DataCollected;
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
