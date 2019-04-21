/** @file configuration_count.h
  * Stop condition based on count of explored configurations.
  */
#pragma once

#include <algorithm>
#include <api/stop_condition/stop_condition.h>

namespace ktt
{

/** @class ConfigurationCount
  * Class which implements stop condition based on count of explored configurations.
  */
class ConfigurationCount : public StopCondition
{
public:
    /** @fn explicit ConfigurationCount(const size_t count)
      * Initializes configuration count condition.
      * @param count Total count of explored configurations which will be tested before condition is satisfied.
      */
    explicit ConfigurationCount(const size_t count) :
        currentCount(0)
    {
        targetCount = std::max(static_cast<size_t>(1), count);
    }

    bool isSatisfied() const override
    {
        return currentCount >= targetCount;
    }

    void initialize(const size_t) override
    {}

    void updateStatus(const bool, const std::vector<ParameterPair>&, const double, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
    {
        currentCount++;
    }
    
    size_t getConfigurationCount() const override
    {
        return targetCount;
    }

    std::string getStatusString() const override
    {
        if (isSatisfied())
        {
            return std::string("Target count of explored configurations reached: " + std::to_string(targetCount));
        }

        return std::string("Current count of explored configurations: " + std::to_string(currentCount) + " / " + std::to_string(targetCount));
    }

private:
    size_t currentCount;
    size_t targetCount;
};

} // namespace ktt
