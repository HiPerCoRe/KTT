#include <Commands/AddKernelCommand.h>

namespace ktt
{

AddKernelCommand::AddKernelCommand(const std::string& name, const std::string& file, const std::vector<std::string>& typeNames) :
    m_Name(name),
    m_File(file),
    m_TypeNames(typeNames)
{}

void AddKernelCommand::Execute(TunerContext& context)
{
    const auto path = context.GetFullPath(m_File);
    const auto definition = context.GetTuner().AddKernelDefinitionFromFile(m_Name, path, m_TypeNames);
    context.SetKernelDefinitionId(definition);
    
    const auto kernel = context.GetTuner().CreateSimpleKernel(m_Name + "_kernel", definition);
    context.SetKernelId(kernel);
}

CommandPriority AddKernelCommand::GetPriority() const
{
    return CommandPriority::Kernel;
}

} // namespace ktt
