#include <Commands/ModifierCommand.h>

namespace ktt
{

ModifierCommand::ModifierCommand(const ModifierType type, const ModifierDimension dimension, const std::string& parameter,
    const ModifierAction action) :
    m_Type(type),
    m_Dimension(dimension),
    m_Parameter(parameter),
    m_Action(action)
{}

void ModifierCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();
    const KernelDefinitionId definitionId = context.GetKernelDefinitionId();
    context.GetTuner().AddThreadModifier(id, {definitionId}, m_Type, m_Dimension, m_Parameter, m_Action);
}

CommandPriority ModifierCommand::GetPriority() const
{
    return CommandPriority::ModifierDefinition;
}

} // namespace ktt
