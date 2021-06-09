/** @file ConfigurationCount.h
  * Stop condition based on count of explored configurations.
  */
#pragma once

#include <cstdint>

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class ConfigurationCount
  * Class which implements stop condition based on count of explored configurations.
  */
class KTT_API ConfigurationCount : public StopCondition
{
public:
    /** @fn explicit ConfigurationCount(const uint64_t count)
      * Initializes configuration count condition.
      * @param count Total count of explored configurations which will be tested before condition is fulfilled.
      */
    explicit ConfigurationCount(const uint64_t count);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    uint64_t m_CurrentCount;
    uint64_t m_TargetCount;
};

} // namespace ktt
