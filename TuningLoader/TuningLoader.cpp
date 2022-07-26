#include <algorithm>
#include <filesystem>
#include <fstream>

#include <Api/KttException.h>
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
    DeserializeCommands(inputStream);

    std::stable_sort(m_Commands.begin(), m_Commands.end(), [](const auto& left, const auto& right)
    {
        return static_cast<int>(left->GetPriority()) < static_cast<int>(right->GetPriority());
    });
}

void TuningLoader::ExecuteCommands()
{
    for (const auto& command : m_Commands)
    {
        command->Execute(*m_Context);
    }
}

void TuningLoader::DeserializeCommands(std::istream& stream)
{
    json input;
    stream >> input;

    if (input.contains("ConfigurationSpace"))
    {
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
    }

    if (input.contains("General"))
    {
        auto& general = input["General"];

        if (general.contains("TimeUnit"))
        {
            auto timeUnitCommand = general.get<TimeUnitCommand>();
            m_Commands.push_back(std::make_unique<TimeUnitCommand>(timeUnitCommand));
        }

        if (general.contains("CompilerOptions"))
        {
            auto compilerOptionsCommand = general.get<CompilerOptionsCommand>();
            m_Commands.push_back(std::make_unique<CompilerOptionsCommand>(compilerOptionsCommand));
        }

        if (general.contains("OutputFile"))
        {
            auto outputCommand = general.get<OutputCommand>();
            m_Commands.push_back(std::make_unique<OutputCommand>(outputCommand));
        }
    }

    if (!input.contains("KernelSpecification"))
    {
        throw KttException("The provided tuning file does not contain kernel specification, so tuning cannot be launched");
    }

    auto& kernelSpecification = input["KernelSpecification"];
    auto createTunerCommand = kernelSpecification.get<CreateTunerCommand>();
    m_Commands.push_back(std::make_unique<CreateTunerCommand>(createTunerCommand));

    auto addKernelCommand = kernelSpecification.get<AddKernelCommand>();
    m_Commands.push_back(std::make_unique<AddKernelCommand>(addKernelCommand));

    if (kernelSpecification.contains("GlobalSize"))
    {
        auto modifierCommand = kernelSpecification["GlobalSize"].get<ModifierCommand>();
        modifierCommand.SetType(ModifierType::Global);
        m_Commands.push_back(std::make_unique<ModifierCommand>(modifierCommand));
    }

    if (kernelSpecification.contains("LocalSize"))
    {
        auto modifierCommand = kernelSpecification["LocalSize"].get<ModifierCommand>();
        modifierCommand.SetType(ModifierType::Local);
        m_Commands.push_back(std::make_unique<ModifierCommand>(modifierCommand));
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

    if (kernelSpecification.contains("SharedMemory"))
    {
        const auto sharedMemoryCommand = kernelSpecification.get<SharedMemoryCommand>();
        m_Commands.push_back(std::make_unique<SharedMemoryCommand>(sharedMemoryCommand));
    }

    m_Commands.push_back(std::make_unique<TuneCommand>());
    Tuner::SetLoggingLevel(LoggingLevel::Debug);
}

} // namespace ktt
