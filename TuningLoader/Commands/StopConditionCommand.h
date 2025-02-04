#pragma once

#include <vector>

#include <Deserialization/StopConditionType.h>
#include <TunerCommand.h>

namespace ktt
{

class StopConditionCommand : public TunerCommand
{
public:
    StopConditionCommand() = default;
    explicit StopConditionCommand(const std::vector<StopConditionType>& types, const std::vector<double>& budgets);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::vector<StopConditionType> m_Types;
    std::vector<double> m_Budgets;
};

} // namespace ktt
