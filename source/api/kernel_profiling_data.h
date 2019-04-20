/** @file kernel_profiling_data.h
  * Class holding profiling information about specific kernel configuration.
  */
#pragma once

#include <vector>
#include <api/kernel_profiling_counter.h>
#include <ktt_platform.h>

namespace ktt
{

/** @class KernelProfilingData
  * Class which holds profiling information about specific kernel configuration.
  */
class KTT_API KernelProfilingData
{
public:
    /** @fn KernelProfilingData()
      * Default constructor, sets validity flag to false.
      */
    KernelProfilingData();

    /** @fn explicit KernelProfilingData(const uint64_t remainingProfilingRuns)
      * Constructor which sets validity flag to false and initializes number of remaining kernel runs needed to gather valid profiling counters.
      * @param remainingProfilingRuns Number of remaining kernel runs needed to gather valid profiling counters.
      */
    explicit KernelProfilingData(const uint64_t remainingProfilingRuns);

    /** @fn explicit KernelProfilingData(const std::vector<KernelProfilingCounter>& profilingCounters)
      * Constructor which sets validity flag to true and fills the structure with provided profiling counters.
      * @param profilingCounters Source vector for profiling counters.
      */
    explicit KernelProfilingData(const std::vector<KernelProfilingCounter>& profilingCounters);

    /** @fn void addCounter(const KernelProfilingCounter& counter)
      * Adds new profiling counter to the vector of existing profiling counters.
      * @param counter Counter which will be added to the vector of existing profiling counters.
      */
    void addCounter(const KernelProfilingCounter& counter);

    /** @fn bool hasCounter(const std::string& counterName)
      * Checks whether profiling counter with specified name exists.
      * @param counterName Name of a profiling counter whose existence will be checked.
      */
    bool hasCounter(const std::string& counterName);

    /** @fn const KernelProfilingCounter& getCounter(const std::string& counterName) const
      * Retrieves profiling counter with specified name. Throws an exception if no corresponding counter is found.
      * @param counterName Name of a profiling counter which will be retrieved.
      * @return Profiling counter with specified name. See KernelProfilingCounter for more information.
      */
    const KernelProfilingCounter& getCounter(const std::string& counterName) const;

    /** @fn uint64_t getRemainingProfilingRuns() const
      * Retrieves number of remaining kernel runs needed to gather valid profiling counters.
      * @return Number of remaining kernel runs needed to gather valid profiling counters.
      */
    uint64_t getRemainingProfilingRuns() const;

    /** @fn const std::vector<KernelProfilingCounter>& getAllCounters() const
      * Retrieves vector of all profiling counters.
      * @return Vector of all profiling counters. See KernelProfilingCounter for more information.
      */
    const std::vector<KernelProfilingCounter>& getAllCounters() const;

    /** @fn bool isValid() const
      * Checks whether the profiling information is valid.
      * @return True if profiling information is valid. False otherwise.
      */
    bool isValid() const;

private:
    std::vector<KernelProfilingCounter> profilingCounters;
    uint64_t remainingProfilingRuns;
    bool validFlag;
};

} // namespace ktt
