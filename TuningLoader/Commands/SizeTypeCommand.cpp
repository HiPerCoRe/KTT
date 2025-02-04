#include <Commands/SizeTypeCommand.h>

namespace ktt
{

SizeTypeCommand::SizeTypeCommand(const GlobalSizeType sizeType) :
    m_SizeType(sizeType)
{}

void SizeTypeCommand::Execute(TunerContext& context)
{
    context.GetTuner().SetGlobalSizeType(m_SizeType);
}

CommandPriority SizeTypeCommand::GetPriority() const
{
    return CommandPriority::General;
}

} // namespace ktt
