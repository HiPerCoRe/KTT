#include <TuningLoader/Commands/TuneCommand.h>

namespace ktt
{

void TuneCommand::Execute(TunerContext& context)
{
    const auto id = context.GetKernelId();
    auto results = context.GetTuner().Tune(id);
    context.SetResults(results);
}

CommandPriority TuneCommand::GetPriority() const
{
    return CommandPriority::Tuning;
}

} // namespace ktt
