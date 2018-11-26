/** @file kernel_profiling_data.h
  * Structure holding information about single profiling counter.
  */
#pragma once

#include <cstdint>
#include <string>
#include <enum/profiling_counter_type.h>
#include <ktt_platform.h>

namespace ktt
{

union KTT_API ProfilingCounterValue
{
    int64_t intValue;
    uint64_t uintValue;
    double doubleValue;
    double percentValue;
    uint64_t throughputValue;
    uint32_t utilizationLevelValue;
};

class KTT_API KernelProfilingCounter
{
public:
    KernelProfilingCounter(const std::string& name, const ProfilingCounterValue& value, const ProfilingCounterType type);

    const std::string& getName() const;
    const ProfilingCounterValue& getValue() const;
    ProfilingCounterType getType() const;

private:
    std::string name;
    ProfilingCounterValue value;
    ProfilingCounterType type;
};

} // namespace ktt
