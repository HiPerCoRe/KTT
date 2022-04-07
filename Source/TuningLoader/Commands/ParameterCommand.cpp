#include <TuningLoader/Commands/ParameterCommand.h>

namespace ktt
{

ParameterCommand::ParameterCommand(const std::string& name, const std::vector<ParameterValue>& values) :
    m_Name(name),
    m_Values(values)
{}

void ParameterCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();
    context.GetTuner().AddParameter(id, m_Name, m_Values);
}

CommandPriority ParameterCommand::GetPriority() const
{
    return CommandPriority::ParameterDefinition;
}

} // namespace ktt
