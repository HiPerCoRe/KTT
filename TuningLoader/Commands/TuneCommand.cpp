#include <Commands/TuneCommand.h>
#include <Utility/FileSystem.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Logger/LoggingLevel.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

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
        const auto input = tuner.LoadResults(filePath, DetectOutputFormat(filePath));
        results = tuner.SimulateTuning(id, input, context.RetrieveStopCondition());
    }

    context.SetResults(results);
}

CommandPriority TuneCommand::GetPriority() const
{
    return CommandPriority::Tuning;
}

OutputFormat TuneCommand::DetectOutputFormat(std::string filePath)
{
    const std::string file = filePath + ".json";
    std::ifstream inputStream(file);

    if (!inputStream.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

	std::ostringstream preview;
    std::string line;
    for (int i = 0; i < 10 && std::getline(inputStream, line); ++i) {
        preview << line << '\n';
    }

    std::string header = preview.str();

    // Look for unique T4 schema keys
    // (replace these with keys guaranteed to appear only in T4 JSON)
    if (header.find("\"metadata\"") != std::string::npos ||
        header.find("\"configuration\"") != std::string::npos ||
        header.find("\"measurements\"") != std::string::npos) {
        return OutputFormat::JSON_T4;
    }

    // Legacy schema identifiers
    if (header.find("\"KttVersion\"") != std::string::npos ||
        header.find("\"ComputeApi\"") != std::string::npos) {
        return OutputFormat::JSON;
    }

    return OutputFormat::JSON;
}
} // namespace ktt
