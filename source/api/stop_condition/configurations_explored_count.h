/** @file configurations_explored_count.h
  * ...
  */
#pragma once

#include <algorithm>
#include "stop_condition.h"

namespace ktt
{

/** @class ConfigurationsExploredCount
  * ...
  */
class ConfigurationsExploredCount : public StopCondition
{
public:
    /** @fn explicit ConfigurationsExploredCount(const size_t count)
      * ...
      */
    explicit ConfigurationsExploredCount(const size_t count) :
        currentCount(0)
    {
        targetCount = std::max(static_cast<size_t>(1), count);
    }

    bool isMet() const override
    {
        return currentCount >= targetCount;
    }

    void initialize(const size_t) override
    {}

    void updateStatus(const double) override
    {
        currentCount++;
    }
    
    std::string getStatusString() const override
    {
        if (isMet())
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
