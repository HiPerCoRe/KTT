#include <Commands/StopConditionCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

StopConditionCommand::StopConditionCommand(const StopConditionType type, const double budgetValue) :
    m_Type(type),
    m_BudgetValue(budgetValue)
{}

void StopConditionCommand::Execute(TunerContext& context)
{
    std::unique_ptr<StopCondition> condition;

    switch (m_Type)
    {
    case StopConditionType::TuningDuration:
        condition = std::make_unique<TuningDuration>(m_BudgetValue);
        break;
    case StopConditionType::ConfigurationCount:
        condition = std::make_unique<ConfigurationCount>(static_cast<uint64_t>(m_BudgetValue));
        break;
    case StopConditionType::ConfigurationFraction:
        condition = std::make_unique<ConfigurationFraction>(m_BudgetValue);
        break;
    default:
        KttLoaderError("Unhandled stop condition type");
    }

    context.SetStopCondition(std::move(condition));
}

CommandPriority StopConditionCommand::GetPriority() const
{
    return CommandPriority::StopCondition;
}

} // namespace ktt
