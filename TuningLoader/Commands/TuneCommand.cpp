#include <Commands/TuneCommand.h>

namespace ktt
{

TuneCommand::TuneCommand(const std::string& simulationInput) :
    m_SimulationInput(simulationInput)
{}

void TuneCommand::Execute(TunerContext& context)
{
    const auto id = context.GetKernelId();
    auto& tuner = context.GetTuner();
    std::vector<KernelResult> results;

    if (m_SimulationInput.empty())
    {
        results = tuner.Tune(id, context.RetrieveStopCondition());
    }
    else
    {
        const auto filePath = context.GetFullPath(m_SimulationInput);
        const auto input = tuner.LoadResults(filePath, OutputFormat::JSON);
        results = tuner.SimulateTuning(id, input, context.RetrieveStopCondition());
    }

    context.SetResults(results);
}

CommandPriority TuneCommand::GetPriority() const
{
    return CommandPriority::Tuning;
}

} // namespace ktt
