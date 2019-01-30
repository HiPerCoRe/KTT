/** @file configuration_fraction.h
  * Stop condition based on fraction of explored configurations.
  */
#pragma once

#include <algorithm>
#include <api/stop_condition/stop_condition.h>

namespace ktt
{

/** @class ConfigurationFraction
  * Class which implements stop condition based on fraction of explored configurations.
  */
class ConfigurationFraction : public StopCondition
{
public:
    /** @fn explicit ConfigurationFraction(const double fraction)
      * Initializes configuration fraction condition.
      * @param fraction Fraction of configurations which will be tested before condition is satisfied. Valid range of values is 0.0 - 1.0
      * corresponding to 0% - 100% of configurations.
      */
    explicit ConfigurationFraction(const double fraction) :
        currentCount(0)
    {
        targetFraction = std::max(0.0, std::min(1.0, fraction));
    }

    bool isSatisfied() const override
    {
        return currentCount >= getConfigurationCount();
    }

    void initialize(const size_t totalConfigurationCount) override
    {
        totalCount = std::max(static_cast<size_t>(1), totalConfigurationCount);
    }

    void updateStatus(const bool, const std::vector<ParameterPair>&, const double, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
    {
        currentCount++;
    }
    
    size_t getConfigurationCount() const override
    {
        return std::max(static_cast<size_t>(1), std::min(totalCount, static_cast<size_t>(totalCount * targetFraction)));
    }

    std::string getStatusString() const override
    {
        if (isSatisfied())
        {
            return std::string("Target count of explored configurations reached: " + std::to_string(getConfigurationCount()));
        }

        return std::string("Current count of explored configurations: " + std::to_string(currentCount) + " / "
            + std::to_string(getConfigurationCount()));
    }

private:
    size_t currentCount;
    size_t totalCount;
    double targetFraction;
};

} // namespace ktt
