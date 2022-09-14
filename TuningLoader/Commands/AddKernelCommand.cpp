#include <filesystem>

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
    std::filesystem::path path(context.GetWorkingDirectory());
    path.append(m_File);

    const auto definition = context.GetTuner().AddKernelDefinitionFromFile(m_Name, path.string(), m_TypeNames);
    context.SetKernelDefinitionId(definition);
    
    const auto kernel = context.GetTuner().CreateSimpleKernel(m_Name + "_kernel", definition);
    context.SetKernelId(kernel);
}

CommandPriority AddKernelCommand::GetPriority() const
{
    return CommandPriority::KernelCreation;
}

} // namespace ktt
