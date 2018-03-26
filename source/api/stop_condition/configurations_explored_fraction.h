/** @file configurations_explored_fraction.h
  * ...
  */
#pragma once

#include <algorithm>
#include "stop_condition.h"

namespace ktt
{

/** @class ConfigurationsExploredFraction
  * ...
  */
class ConfigurationsExploredFraction : public StopCondition
{
public:
    /** @fn explicit ConfigurationsExploredFraction(const double fraction)
      * ...
      */
    explicit ConfigurationsExploredFraction(const double fraction) :
        currentCount(0)
    {
        targetFraction = std::max(0.0, std::min(1.0, fraction));
    }

    bool isMet() const override
    {
        return (currentCount / static_cast<double>(totalCount)) >= targetFraction;
    }

    void initialize(const size_t totalConfigurationCount) override
    {
        totalCount = std::max(static_cast<size_t>(1), totalConfigurationCount);
    }

    void updateStatus(const double) override
    {
        currentCount++;
    }
    
    std::string getStatusString() const override
    {
        if (isMet())
        {
            return std::string("Target fraction of explored configurations reached: " + std::to_string(targetFraction * 100.0) + "%");
        }

        return std::string("Current fraction of explored configurations: " + std::to_string(currentCount / totalCount * 100.0) + "% / "
            + std::to_string(targetFraction * 100.0) + "%");
    }

private:
    size_t currentCount;
    size_t totalCount;
    double targetFraction;
};

} // namespace ktt
