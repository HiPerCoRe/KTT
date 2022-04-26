#include <TuningLoader/Commands/TimeUnitCommand.h>

namespace ktt
{

TimeUnitCommand::TimeUnitCommand(const TimeUnit timeUnit) :
    m_TimeUnit(timeUnit)
{}

void TimeUnitCommand::Execute(TunerContext& context)
{
    context.GetTuner().SetTimeUnit(m_TimeUnit);
}

CommandPriority TimeUnitCommand::GetPriority() const
{
    return CommandPriority::General;
}

} // namespace ktt
