#include <algorithm>
#include <filesystem>
#include <fstream>

#include <Api/KttException.h>
#include <Commands/AddArgumentCommand.h>
#include <Commands/AddKernelCommand.h>
#include <Commands/CompilerOptionsCommand.h>
#include <Commands/ConstraintCommand.h>
#include <Commands/CreateTunerCommand.h>
#include <Commands/ModifierCommand.h>
#include <Commands/OutputCommand.h>
#include <Commands/ParameterCommand.h>
#include <Commands/TimeUnitCommand.h>
#include <Commands/TuneCommand.h>
#include <JsonCommandConverters.h>
#include <TunerCommand.h>
#include <TunerContext.h>
#include <TuningLoader.h>

namespace ktt
{

TuningLoader::TuningLoader() :
    m_Context(std::make_unique<TunerContext>())
{}

TuningLoader::~TuningLoader() = default;

void TuningLoader::LoadTuningFile(const std::string& file)
{
    std::ifstream inputStream(file);

    if (!inputStream.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

    const std::string directory = std::filesystem::path(file).parent_path().string();
    m_Context->SetWorkingDirectory(directory);

    json input;
    inputStream >> input;

    auto& configurationSpace = input["ConfigurationSpace"];

    if (configurationSpace.contains("TuningParameters"))
    {
        const auto parameterCommands = configurationSpace["TuningParameters"].get<std::vector<ParameterCommand>>();

        for (const auto& command : parameterCommands)
        {
            m_Commands.push_back(std::make_unique<ParameterCommand>(command));
        }
    }

    if (configurationSpace.contains("Conditions"))
    {
        const auto constraintCommands = configurationSpace["Conditions"].get<std::vector<ConstraintCommand>>();

        for (const auto& command : constraintCommands)
        {
            m_Commands.push_back(std::make_unique<ConstraintCommand>(command));
        }
    }

    if (input.contains("General"))
    {
        auto& general = input["General"];

        if (input.contains("TimeUnit"))
        {
            auto timeUnitCommand = general.get<TimeUnitCommand>();
            m_Commands.push_back(std::make_unique<TimeUnitCommand>(timeUnitCommand));
        }

        if (input.contains("CompilerOptions"))
        {
            auto compilerOptionsCommand = general.get<CompilerOptionsCommand>();
            m_Commands.push_back(std::make_unique<CompilerOptionsCommand>(compilerOptionsCommand));
        }

        if (input.contains("OutputFile"))
        {
            auto outputCommand = general.get<OutputCommand>();
            m_Commands.push_back(std::make_unique<OutputCommand>(outputCommand));
        }
    }

    auto& kernelSpecification = input["KernelSpecification"];
    auto createTunerCommand = kernelSpecification.get<CreateTunerCommand>();
    m_Commands.push_back(std::make_unique<CreateTunerCommand>(createTunerCommand));

    auto addKernelCommand = kernelSpecification.get<AddKernelCommand>();
    m_Commands.push_back(std::make_unique<AddKernelCommand>(addKernelCommand));

    if (kernelSpecification.contains("Modifiers"))
    {
        const auto modifierCommands = kernelSpecification["Modifiers"].get<std::vector<ModifierCommand>>();

        for (const auto& command : modifierCommands)
        {
            m_Commands.push_back(std::make_unique<ModifierCommand>(command));
        }
    }

    if (kernelSpecification.contains("Arguments"))
    {
        auto argumentCommands = kernelSpecification["Arguments"].get<std::vector<AddArgumentCommand>>();

        std::sort(argumentCommands.begin(), argumentCommands.end(), [](const auto& left, const auto& right)
        {
            return left.GetOrder() < right.GetOrder();
        });

        for (const auto& command : argumentCommands)
        {
            m_Commands.push_back(std::make_unique<AddArgumentCommand>(command));
        }
    }

    m_Commands.push_back(std::make_unique<TuneCommand>());
}

void TuningLoader::ExecuteCommands()
{
    std::stable_sort(m_Commands.begin(), m_Commands.end(), [](const auto& left, const auto& right)
    {
        return static_cast<int>(left->GetPriority()) < static_cast<int>(right->GetPriority());
    });

    for (const auto& command : m_Commands)
    {
        command->Execute(*m_Context);
    }
}

} // namespace ktt
