#include <Commands/CompilerOptionsCommand.h>

namespace ktt
{

CompilerOptionsCommand::CompilerOptionsCommand(const std::vector<std::string>& options) :
    m_Options(options)
{}

void CompilerOptionsCommand::Execute(TunerContext& context)
{
    std::string finalOptions;

    for (const auto& option : m_Options)
    {
        finalOptions += option;

        if (&option != &m_Options.back())
        {
            finalOptions += " ";
        }
    }

    context.GetTuner().SetCompilerOptions(finalOptions);
}

CommandPriority CompilerOptionsCommand::GetPriority() const
{
    return CommandPriority::General;
}

} // namespace ktt
