/** @file ConfigurationFraction.h
  * Stop condition based on fraction of explored configurations.
  */
#pragma once

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class ConfigurationFraction
  * Class which implements stop condition based on fraction of explored configurations.
  */
class KTT_API ConfigurationFraction : public StopCondition
{
public:
    /** @fn explicit ConfigurationFraction(const double fraction)
      * Initializes configuration fraction condition.
      * @param fraction Fraction of configurations which will be tested before condition is fulfilled. The valid range of values
      * is 0.0 - 1.0 corresponding to 0% - 100% of configurations.
      */
    explicit ConfigurationFraction(const double fraction);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    uint64_t m_CurrentCount;
    uint64_t m_TotalCount;
    double m_TargetFraction;

    double GetExploredFraction() const;
};

} // namespace ktt
