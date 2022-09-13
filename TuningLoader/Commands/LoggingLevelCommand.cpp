#include <Commands/LoggingLevelCommand.h>

namespace ktt
{

LoggingLevelCommand::LoggingLevelCommand(const LoggingLevel level) :
    m_LoggingLevel(level)
{}

void LoggingLevelCommand::Execute([[maybe_unused]] TunerContext& context)
{
    Tuner::SetLoggingLevel(m_LoggingLevel);
}

CommandPriority LoggingLevelCommand::GetPriority() const
{
    return CommandPriority::Logging;
}

} // namespace ktt
