/** @file KernelProfilingData.h
  * Profiling information about a kernel run under specific configuration.
  */
#pragma once

#include <vector>

#include <Api/Output/KernelProfilingCounter.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class KernelProfilingData
  * Class which holds profiling information about a kernel run under specific configuration.
  */
class KTT_API KernelProfilingData
{
public:
    /** @fn KernelProfilingData() = default
      * Constructor which creates empty invalid profiling data.
      */
    KernelProfilingData() = default;

    /** @fn explicit KernelProfilingData(const uint64_t remainingRuns)
      * Constructor which creates invalid profiling data and initializes number of remaining kernel runs needed to gather
      * valid profiling counters.
      * @param remainingRuns Number of remaining kernel runs needed to gather valid profiling counters.
      */
    explicit KernelProfilingData(const uint64_t remainingRuns);

    /** @fn explicit KernelProfilingData(const std::vector<KernelProfilingCounter>& counters)
      * Constructor which creates valid profiling data and fills the structure with provided profiling counters.
      * @param counters Vector of profiling counters.
      */
    explicit KernelProfilingData(const std::vector<KernelProfilingCounter>& counters);

    /** @fn bool IsValid() const
      * Checks whether the profiling information is valid.
      * @return True if profiling information is valid. False otherwise.
      */
    bool IsValid() const;

    /** @fn bool HasCounter(const std::string& name) const
      * Checks whether profiling counter with specified name exists.
      * @param name Name of a profiling counter whose existence will be checked.
      */
    bool HasCounter(const std::string& name) const;

    /** @fn const KernelProfilingCounter& GetCounter(const std::string& name) const
      * Retrieves profiling counter with specified name. Throws an exception if no corresponding counter is found.
      * @param name Name of a profiling counter which will be retrieved.
      * @return Profiling counter with specified name. See KernelProfilingCounter for more information.
      */
    const KernelProfilingCounter& GetCounter(const std::string& name) const;

    /** @fn const std::vector<KernelProfilingCounter>& GetCounters() const
      * Retrieves vector of all profiling counters.
      * @return Vector of all profiling counters. See KernelProfilingCounter for more information.
      */
    const std::vector<KernelProfilingCounter>& GetCounters() const;

    /** @fn void SetCounters(const std::vector<KernelProfilingCounter>& counters)
      * Fills the structure with provided profiling counters. Sets number of remaining kernel runs to zero.
      * @param counters Vector of profiling counters.
      */
    void SetCounters(const std::vector<KernelProfilingCounter>& counters);

    /** @fn void AddCounter(const KernelProfilingCounter& counter)
      * Adds new profiling counter to the vector of existing profiling counters.
      * @param counter Counter which will be added to the vector of existing profiling counters.
      */
    void AddCounter(const KernelProfilingCounter& counter);

    /** @fn bool HasRemainingProfilingRuns() const
      * Checks whether there are more kernel runs needed to gather valid profiling counters.
      * @return Information whether there are more kernel runs needed to gather valid profiling counters.
      */
    bool HasRemainingProfilingRuns() const;

    /** @fn uint64_t GetRemainingProfilingRuns() const
      * Retrieves number of remaining kernel runs needed to gather valid profiling counters.
      * Note that this method currently returns incorrect information when using new CUPTI profiling API.
      * @return Number of remaining kernel runs needed to gather valid profiling counters.
      */
    uint64_t GetRemainingProfilingRuns() const;

    /** @fn void DecreaseRemainingProfilingRuns()
      * Decreases number of remaining kernel runs needed to gather valid profiling counters. Performs no operation
      * if profiling data is valid.
      */
    void DecreaseRemainingProfilingRuns();

private:
    std::vector<KernelProfilingCounter> m_Counters;
    uint64_t m_RemainingRuns;
};

} // namespace ktt
