/** @file FailureFraction.h
  * Stop condition based on fraction of failed kernel runs.
  */
#pragma once

#include <cstdint>

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class FailureFraction
  * Class which implements stop condition based on fraction of failed kernel runs.
  */
class KTT_API FailureFraction : public StopCondition
{
public:
    /** @fn explicit FailureFraction(const double fraction, const uint64_t minCount = 0)
      * Initializes failure fraction condition.
      * @param fraction Fraction of failed kernel runs before condition is fulfilled. The valid range of values
      * is 0.0 - 1.0 corresponding to 0% - 100% of explored configurations.
      * @param minCount The number of configurations that is always explored
      */
    explicit FailureFraction(const double fraction, const uint64_t minCount = 0);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    uint64_t m_TotalExplored;
    uint64_t m_Failures;
    double m_TargetFraction;
    uint_fast64_t m_MinCount;

    double GetFailureFraction() const;
};

} // namespace ktt