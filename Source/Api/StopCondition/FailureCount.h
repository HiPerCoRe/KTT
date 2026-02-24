/** @file FailureCount.h
  * Stop condition based on count of failed kernel runs.
  */
#pragma once

#include <cstdint>

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class FailureCount
  * Class which implements stop condition based on count of failed kernel runs.
  */
class KTT_API FailureCount : public StopCondition
{
public:
    /** @fn explicit FailureCount(const uint64_t maxFailures)
      * Initializes failure count condition.
      * @param maxFailures Maximum number of failed kernel runs before condition is fulfilled.
      */
    explicit FailureCount(const uint64_t maxFailures);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    uint64_t m_CurrentFailures;
    uint64_t m_MaxFailures;
};

} // namespace ktt