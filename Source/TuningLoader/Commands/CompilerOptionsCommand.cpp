#include <TuningLoader/Commands/CompilerOptionsCommand.h>

namespace ktt
{

CompilerOptionsCommand::CompilerOptionsCommand(const std::string& options) :
    m_Options(options)
{}

void CompilerOptionsCommand::Execute(TunerContext& context)
{
    context.GetTuner().SetCompilerOptions(m_Options);
}

CommandPriority CompilerOptionsCommand::GetPriority() const
{
    return CommandPriority::General;
}

} // namespace ktt
