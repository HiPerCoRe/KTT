#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <api/kernel_profiling_counter.h>

namespace ktt
{

struct CUPTIMetric
{
public:
    CUPTIMetric() :
        name(""),
        rangeCount(0)
    {}

    KernelProfilingCounter getCounter() const
    {
        if (rangeToMetricValue.empty())
        {
            throw std::runtime_error("Unable to retrieve profiling counter for uninitialized metric");
        }

        ProfilingCounterValue value;

        for (const auto& pair : rangeToMetricValue)
        {
            value.doubleValue = pair.second;
            break;
        }

        return KernelProfilingCounter(name, value, ProfilingCounterType::Double);
    }

    std::string name;
    size_t rangeCount;
    std::map<std::string, double> rangeToMetricValue;
};

} // namespace ktt
