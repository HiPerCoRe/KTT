#include <Commands/StopConditionCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

StopConditionCommand::StopConditionCommand(const std::vector<StopConditionType>& types, const std::vector<double>& budgets) :
    m_Types(types),
    m_Budgets(budgets)
{}

void StopConditionCommand::Execute(TunerContext& context)
{
    std::vector<std::shared_ptr<StopCondition>> conditions;

    for (size_t i = 0; i < m_Types.size(); ++i)
    {
        std::shared_ptr<StopCondition> condition;

        switch (m_Types[i])
        {
        case StopConditionType::TuningDuration:
            condition = std::make_unique<TuningDuration>(m_Budgets[i]);
            break;
        case StopConditionType::ConfigurationCount:
            condition = std::make_unique<ConfigurationCount>(static_cast<uint64_t>(m_Budgets[i]));
            break;
        case StopConditionType::ConfigurationFraction:
            condition = std::make_unique<ConfigurationFraction>(m_Budgets[i]);
            break;
        default:
            KttLoaderError("Unhandled stop condition type");
        }

        conditions.push_back(condition);
    }

    auto unionCondition = std::make_unique<UnionCondition>(conditions);
    context.SetStopCondition(std::move(unionCondition));
}

CommandPriority StopConditionCommand::GetPriority() const
{
    return CommandPriority::StopCondition;
}

} // namespace ktt
