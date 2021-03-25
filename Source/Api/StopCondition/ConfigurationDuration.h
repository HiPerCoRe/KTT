/** @file ConfigurationDuration.h
  * Stop condition based on computation duration of a configuration.
  */
#pragma once

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class ConfigurationDuration
  * Class which implements stop condition based on computation duration of a configuration.
  */
class KTT_API ConfigurationDuration : public StopCondition
{
public:
    /** @fn explicit ConfigurationDuration(const double duration)
      * Initializes configuration duration condition.
      * @param duration Condition will be fulfilled when configuration with duration below the specified amount is found.
      * The duration is specified in milliseconds.
      */
    explicit ConfigurationDuration(const double duration);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    double m_BestDuration;
    double m_TargetDuration;
};

} // namespace ktt
