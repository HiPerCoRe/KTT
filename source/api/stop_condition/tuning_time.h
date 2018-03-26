/** @file tuning_time.h
  * ...
  */
#pragma once

#include <algorithm>
#include <chrono>
#include "stop_condition.h"

namespace ktt
{

/** @class TuningTime
  * ...
  */
class TuningTime : public StopCondition
{
public:
    /** @fn explicit TuningTime(const double time)
      * ...
      */
    explicit TuningTime(const double time) :
        passedTime(0.0)
    {
        targetTime = std::max(0.0, time);
    }

    bool isMet() const override
    {
        return passedTime > targetTime;
    }

    void initialize(const size_t) override
    {
        initialTime = std::chrono::steady_clock::now();
    }

    void updateStatus(const double) override
    {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        passedTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - initialTime).count()) / 1000.0;
    }
    
    std::string getStatusString() const override
    {
        if (isMet())
        {
            return std::string("Target tuning time reached: " + std::to_string(targetTime) + " seconds");
        }

        return std::string("Current tuning time: " + std::to_string(passedTime) + " / "
            + std::to_string(targetTime) + " seconds");
    }

private:
    std::chrono::steady_clock::time_point initialTime;
    double passedTime;
    double targetTime;
};

} // namespace ktt
