#include <Commands/ValidationCommand.h>

namespace ktt
{

ValidationCommand::ValidationCommand(const ArgumentId& referenceId, const AddArgumentCommand& command, const ValidationMethod validationMethod, const double validationThreshold) :
    m_ReferencedId(referenceId),
    m_ArgumentCommand(command),
    m_ValidationMethod(validationMethod),
    m_ValidationThreshold(validationThreshold)
{}

void ValidationCommand::Execute(TunerContext& context)
{
    m_ArgumentCommand.Execute(context);
    context.GetTuner().SetValidationMethod(m_ValidationMethod, m_ValidationThreshold);
    context.GetTuner().SetReferenceArgument(m_ReferencedId, m_ArgumentCommand.GetId());
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
