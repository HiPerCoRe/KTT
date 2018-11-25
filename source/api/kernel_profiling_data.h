/** @file kernel_profiling_data.h
  * Structure holding profiling information about specific kernel configuration.
  */
#pragma once

#include <vector>
#include <api/kernel_profiling_counter.h>
#include <ktt_platform.h>

namespace ktt
{

class KTT_API KernelProfilingData
{
public:
    KernelProfilingData();
    KernelProfilingData(const std::vector<KernelProfilingCounter>& profilingCounters);

    void addCounter(const KernelProfilingCounter& counter);
    bool hasCounter(const std::string& counterName);
    const KernelProfilingCounter& getCounter(const std::string& counterName) const;
    const std::vector<KernelProfilingCounter>& getAllCounters() const;
    bool isValid() const;

private:
    std::vector<KernelProfilingCounter> profilingCounters;
    bool validFlag;
};

} // namespace ktt
