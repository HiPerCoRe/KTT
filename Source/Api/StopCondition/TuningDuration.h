/** @file TuningDuration.h
  * Stop condition based on total tuning duration.
  */
#pragma once

#include <chrono>

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class TuningDuration
  * Class which implements stop condition based on total tuning duration.
  */
class KTT_API TuningDuration : public StopCondition
{
public:
    /** @fn explicit TuningDuration(const double duration)
      * Initializes tuning duration condition.
      * @param duration Condition will be fulfilled when the specified amount of time has passed. The measurement starts when
      * kernel tuning begins. Duration is specified in seconds.
      */
    explicit TuningDuration(const double duration);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    std::chrono::steady_clock::time_point m_InitialTime;
    double m_PassedTime;
    double m_TargetTime;
};

} // namespace ktt
