#include <Commands/ModifierCommand.h>

namespace ktt
{

ModifierCommand::ModifierCommand(const ModifierType type, const std::map<ModifierDimension, std::string>& scripts) :
    m_Type(type),
    m_Scripts(scripts)
{}

void ModifierCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();
    const KernelDefinitionId definitionId = context.GetKernelDefinitionId();

    for (const auto& script : m_Scripts)
    {
        context.GetTuner().AddScriptThreadModifier(id, {definitionId}, m_Type, script.first, script.second);
    }
}

CommandPriority ModifierCommand::GetPriority() const
{
    return CommandPriority::Modifier;
}

void ModifierCommand::SetType(const ModifierType type)
{
    m_Type = type;
}

} // namespace ktt
