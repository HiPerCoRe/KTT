/** @file configuration_duration.h
  * Stop condition based on computation duration of a configuration.
  */
#pragma once

#include <algorithm>
#include <limits>
#include "stop_condition.h"

namespace ktt
{

/** @class ConfigurationDuration
  * Class which implements stop condition based on computation duration of a configuration.
  */
class ConfigurationDuration : public StopCondition
{
public:
    /** @fn explicit ConfigurationDuration(const double duration)
      * Initializes configuration duration condition.
      * @param duration Condition will be satisfied when configuration with duration below the specified amount is found. The duration is specified
      * in milliseconds.
      */
    explicit ConfigurationDuration(const double duration) :
        bestDuration(std::numeric_limits<double>::max())
    {
        targetDuration = std::max(0.0, duration);
    }

    bool isMet() const override
    {
        return bestDuration <= targetDuration;
    }

    void initialize(const size_t totalConfigurationCount) override
    {
        totalCount = totalConfigurationCount;
    }

    void updateStatus(const double previousConfigurationDuration) override
    {
        bestDuration = std::min(bestDuration, previousConfigurationDuration / 1000000.0);
    }
    
    size_t getConfigurationCount() const override
    {
        return totalCount;
    }

    std::string getStatusString() const override
    {
        if (isMet())
        {
            return std::string("Target configuration duration reached: " + std::to_string(targetDuration) + "ms");
        }

        return std::string("Current best known configuration duration: " + std::to_string(bestDuration) + "ms / " + std::to_string(targetDuration)
            + "ms");
    }

private:
    size_t totalCount;
    double bestDuration;
    double targetDuration;
};

} // namespace ktt
