/** @file configuration_duration.h
  * ...
  */
#pragma once

#include <algorithm>
#include <limits>
#include "stop_condition.h"

namespace ktt
{

/** @class ConfigurationDuration
  * ...
  */
class ConfigurationDuration : public StopCondition
{
public:
    /** @fn explicit ConfigurationDuration(const double duration)
      * ...
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

    void initialize(const size_t) override
    {}

    void updateStatus(const double previousConfigurationDuration) override
    {
        bestDuration = std::min(bestDuration, previousConfigurationDuration / 1'000'000.0);
    }
    
    std::string getStatusString() const override
    {
        if (isMet())
        {
            return std::string("Target configuration duration reached: " + std::to_string(targetDuration) + "ms");
        }

        return std::string("Current configuration duration: " + std::to_string(bestDuration) + "ms / " + std::to_string(targetDuration) + "ms");
    }

private:
    double bestDuration;
    double targetDuration;
};

} // namespace ktt
