/** @file tuning_duration.h
  * Stop condition based on total tuning duration.
  */
#pragma once

#include <algorithm>
#include <chrono>
#include <api/stop_condition/stop_condition.h>

namespace ktt
{

/** @class TuningDuration
  * Class which implements stop condition based on total tuning duration.
  */
class TuningDuration : public StopCondition
{
public:
    /** @fn explicit TuningDuration(const double duration)
      * Initializes tuning duration condition.
      * @param duration Condition will be satisfied when specified amount of time has passed. The measurement starts when kernel tuning begins.
      * Duration is specified in seconds.
      */
    explicit TuningDuration(const double duration) :
        passedTime(0.0)
    {
        targetTime = std::max(0.0, duration);
    }

    bool isSatisfied() const override
    {
        return passedTime > targetTime;
    }

    void initialize(const size_t totalConfigurationCount) override
    {
        totalCount = totalConfigurationCount;
        initialTime = std::chrono::steady_clock::now();
    }

    void updateStatus(const bool, const std::vector<ParameterPair>&, const double, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
    {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        passedTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - initialTime).count()) / 1000.0;
    }
    
    size_t getConfigurationCount() const override
    {
        return totalCount;
    }

    std::string getStatusString() const override
    {
        if (isSatisfied())
        {
            return std::string("Target tuning time reached: " + std::to_string(targetTime) + " seconds");
        }

        return std::string("Current tuning time: " + std::to_string(passedTime) + " / "
            + std::to_string(targetTime) + " seconds");
    }

private:
    size_t totalCount;
    std::chrono::steady_clock::time_point initialTime;
    double passedTime;
    double targetTime;
};

} // namespace ktt
