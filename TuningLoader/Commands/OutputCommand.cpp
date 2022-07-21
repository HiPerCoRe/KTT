#include <Commands/OutputCommand.h>

namespace ktt
{

OutputCommand::OutputCommand(const std::string& file, const OutputFormat format) :
    m_File(file),
    m_Format(format)
{}

void OutputCommand::Execute(TunerContext& context)
{
    context.GetTuner().SaveResults(context.GetResults(), m_File, m_Format);
}

CommandPriority OutputCommand::GetPriority() const
{
    return CommandPriority::Output;
}

} // namespace ktt
