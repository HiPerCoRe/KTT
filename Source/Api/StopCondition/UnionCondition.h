/** @file UnionCondition.h
  * Stop condition which is fulfilled when any of the underlying conditions are met.
  */
#pragma once

#include <memory>
#include <vector>

#include <Api/StopCondition/StopCondition.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class UnionCondition
  * Class which implements stop condition which is fulfilled when any of the underlying conditions are met.
  */
class KTT_API UnionCondition : public StopCondition
{
public:
    /** @fn explicit UnionCondition(std::vector<std::shared_ptr<StopCondition>>& conditions)
      * Initializes union condition.
      * @param conditions Underlying conditions which are all evaluated, the union is fulfilled when at least one condition is met.
      */
    explicit UnionCondition(const std::vector<std::shared_ptr<StopCondition>>& conditions);

    bool IsFulfilled() const override;
    void Initialize(const uint64_t configurationsCount) override;
    void Update(const KernelResult& result) override;
    std::string GetStatusString() const override;

private:
    std::vector<std::shared_ptr<StopCondition>> m_Conditions;
};

} // namespace ktt
