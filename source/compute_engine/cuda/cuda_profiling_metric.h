#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda.h>
#include <cupti.h>

namespace ktt
{

struct CUDAProfilingMetric
{
public:
    CUDAProfilingMetric() :
        metricId(0),
        metricName(""),
        device(0),
        eventGroupSets(nullptr),
        currentSetIndex(0),
        eventCount(0)
    {}

    CUpti_MetricID metricId;
    std::string metricName;
    CUdevice device;
    CUpti_EventGroupSets* eventGroupSets;
    uint32_t currentSetIndex;
    uint32_t eventCount;
    std::vector<CUpti_EventID> eventIds;
    std::vector<uint64_t> eventValues;
    std::vector<bool> eventStatuses;
};

} // namespace ktt
