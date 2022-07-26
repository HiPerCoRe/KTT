#include <filesystem>

#include <Commands/AddKernelCommand.h>

namespace ktt
{

AddKernelCommand::AddKernelCommand(const std::string& name, const std::string& file) :
    m_Name(name),
    m_File(file)
{}

void AddKernelCommand::Execute(TunerContext& context)
{
    std::filesystem::path path(context.GetWorkingDirectory());
    path.append(m_File);

    const auto definition = context.GetTuner().AddKernelDefinitionFromFile(m_Name, path.string());
    context.SetKernelDefinitionId(definition);
    
    const auto kernel = context.GetTuner().CreateSimpleKernel(m_Name + "_kernel", definition);
    context.SetKernelId(kernel);
}

CommandPriority AddKernelCommand::GetPriority() const
{
    return CommandPriority::KernelCreation;
}

} // namespace ktt
