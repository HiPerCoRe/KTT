#include <Commands/ProfilingCommand.h>

namespace ktt
{

ProfilingCommand::ProfilingCommand(const bool profilingOn) :
    m_ProfilingOn(profilingOn)
{}

void ProfilingCommand::Execute(TunerContext& context)
{
    context.GetTuner().SetProfiling(m_ProfilingOn);
}

CommandPriority ProfilingCommand::GetPriority() const
{
    return CommandPriority::General;
}

} // namespace ktt
