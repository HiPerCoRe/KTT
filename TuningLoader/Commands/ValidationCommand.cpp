#include <Commands/ValidationCommand.h>

namespace ktt
{

ValidationCommand::ValidationCommand(const ArgumentId& id, const ArgumentId& referenceId) :
    m_Id(id),
    m_ReferencedId(referenceId)
{}

void ValidationCommand::Execute(TunerContext& context)
{
    context.GetTuner().SetReferenceArgument(m_Id, m_ReferencedId);
}

CommandPriority ValidationCommand::GetPriority() const
{
    return CommandPriority::Validation;
}

} // namespace ktt
