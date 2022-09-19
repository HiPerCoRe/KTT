#pragma once

#include <StopConditionType.h>
#include <TunerCommand.h>

namespace ktt
{

class StopConditionCommand : public TunerCommand
{
public:
    StopConditionCommand() = default;
    explicit StopConditionCommand(const StopConditionType type, const double budgetValue);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    StopConditionType m_Type;
    double m_BudgetValue;
};

} // namespace ktt
