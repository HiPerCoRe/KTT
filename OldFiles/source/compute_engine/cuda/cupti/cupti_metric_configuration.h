#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ktt
{

struct CUPTIMetricConfiguration
{
public:
    std::vector<std::string> metricNames;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> scratchBuffer;
    std::vector<uint8_t> counterDataImage;
    uint32_t maxProfiledRanges;
    bool dataCollected;
};

} // namespace ktt
