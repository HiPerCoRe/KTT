#include <algorithm>
#include <fstream>

#include <Api/KttException.h>
#include <TuningLoader/Commands/AddKernelCommand.h>
#include <TuningLoader/Commands/ConstraintCommand.h>
#include <TuningLoader/Commands/CreateTunerCommand.h>
#include <TuningLoader/Commands/OutputCommand.h>
#include <TuningLoader/Commands/ParameterCommand.h>
#include <TuningLoader/Commands/TimeUnitCommand.h>
#include <TuningLoader/Commands/TuneCommand.h>
#include <TuningLoader/JsonCommandConverters.h>
#include <TuningLoader/TuningLoader.h>

namespace ktt
{

void TuningLoader::LoadTuningDescription(const std::string& file)
{
    std::ifstream inputStream(file);

    if (!inputStream.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

    json input;
    inputStream >> input;

    auto& configurationSpace = input["ConfigurationSpace"];
    const auto parameterCommands = configurationSpace["TuningParameters"].get<std::vector<ParameterCommand>>();

    for (const auto& command : parameterCommands)
    {
        m_Commands.push_back(std::make_unique<ParameterCommand>(command));
    }

    const auto constraintCommands = configurationSpace["Conditions"].get<std::vector<ConstraintCommand>>();

    for (const auto& command : constraintCommands)
    {
        m_Commands.push_back(std::make_unique<ConstraintCommand>(command));
    }

    auto& general = input["General"];
    auto timeUnitCommand = general.get<TimeUnitCommand>();
    m_Commands.push_back(std::make_unique<TimeUnitCommand>(timeUnitCommand));

    auto outputCommand = general.get<OutputCommand>();
    m_Commands.push_back(std::make_unique<OutputCommand>(outputCommand));

    auto& kernelSpecification = input["KernelSpecification"];
    auto createTunerCommand = kernelSpecification.get<CreateTunerCommand>();
    m_Commands.push_back(std::make_unique<CreateTunerCommand>(createTunerCommand));

    auto addKernelCommand = kernelSpecification.get<AddKernelCommand>();
    m_Commands.push_back(std::make_unique<AddKernelCommand>(addKernelCommand));

    m_Commands.push_back(std::make_unique<TuneCommand>());
}

void TuningLoader::ApplyCommands()
{
    std::sort(m_Commands.begin(), m_Commands.end(), [](const auto& left, const auto& right)
    {
        return static_cast<int>(left->GetPriority()) < static_cast<int>(right->GetPriority());
    });

    for (const auto& command : m_Commands)
    {
        command->Execute(m_Context);
    }
}

} // namespace ktt
