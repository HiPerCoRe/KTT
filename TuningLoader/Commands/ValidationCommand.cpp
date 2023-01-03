#include <Commands/ValidationCommand.h>

namespace ktt
{

ValidationCommand::ValidationCommand(const ArgumentId& referenceId, const AddArgumentCommand& command) :
    m_ReferencedId(referenceId),
    m_ArgumentCommand(command)
{}

void ValidationCommand::Execute(TunerContext& context)
{
    m_ArgumentCommand.Execute(context);
    context.GetTuner().SetReferenceArgument(m_ArgumentCommand.GetId(), m_ReferencedId);
}

CommandPriority ValidationCommand::GetPriority() const
{
    return CommandPriority::Validation;
}

void ValidationCommand::SetReferenceProperties(const std::vector<AddArgumentCommand>& commands)
{
    for (const auto& command : commands)
    {
        if (command.GetId() == m_ReferencedId)
        {
            m_ArgumentCommand.SetReferenceProperties(command);
            break;
        }
    }
}

} // namespace ktt
