#include <TuningLoader/Commands/ConstraintCommand.h>

namespace ktt
{

ConstraintCommand::ConstraintCommand(const std::vector<std::string>& parameters, const std::string& expression) :
    m_Parameters(parameters),
    m_Expression(expression)
{}

void ConstraintCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();
    context.GetTuner().AddScriptConstraint(id, m_Parameters, m_Expression);
}

CommandPriority ConstraintCommand::GetPriority() const
{
    return CommandPriority::ConstraintDefinition;
}

} // namespace ktt
