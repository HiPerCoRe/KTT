#include <TuningLoader/Commands/AddKernelCommand.h>

namespace ktt
{

AddKernelCommand::AddKernelCommand(const std::string& name, const std::string& source) :
    m_Name(name),
    m_Source(source)
{}

void AddKernelCommand::Execute(TunerContext& context)
{
    const auto definition = context.GetTuner().AddKernelDefinition(m_Name, m_Source, DimensionVector(), DimensionVector());
    context.SetKernelDefinitionId(definition);
    
    const auto kernel = context.GetTuner().CreateSimpleKernel(m_Name + "_kernel", definition);
    context.SetKernelId(kernel);
}

CommandPriority AddKernelCommand::GetPriority() const
{
    return CommandPriority::KernelCreation;
}

} // namespace ktt