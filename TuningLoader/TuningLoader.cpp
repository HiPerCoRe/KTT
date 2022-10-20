#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <json-schema.hpp>

#include <Api/KttException.h>
#include <Commands/TuneCommand.h>
#include <Deserialization/JsonCommandConverters.h>
#include <TunerCommand.h>
#include <TunerContext.h>
#include <TuningLoader.h>
#include <TuningSchema.h>

namespace ktt
{

using nlohmann::json_schema::json_validator;

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

    std::stringstream stream;
    stream << inputStream.rdbuf();
    const std::string tuningScript = stream.str();

    if (!ValidateFormat(tuningScript))
    {
        throw KttException("Tuning file validation failed");
    }

    DeserializeCommands(tuningScript);

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

void TuningLoader::DeserializeCommands(const std::string& tuningScript)
{
    json input = json::parse(tuningScript);

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

    if (input.contains("Budget"))
    {
        auto& budget = input["Budget"];
        auto stopConditionCommand = budget.get<StopConditionCommand>();
        m_Commands.push_back(std::make_unique<StopConditionCommand>(stopConditionCommand));
    }

    if (input.contains("Search"))
    {
        auto& search = input["Search"];
        auto searcherCommand = search.get<SearcherCommand>();
        m_Commands.push_back(std::make_unique<SearcherCommand>(searcherCommand));
    }

    if (input.contains("General"))
    {
        auto& general = input["General"];

        if (general.contains("LoggingLevel"))
        {
            auto loggingLevelCommand = general.get<LoggingLevelCommand>();
            m_Commands.push_back(std::make_unique<LoggingLevelCommand>(loggingLevelCommand));
        }

        if (general.contains("TimeUnit"))
        {
            auto timeUnitCommand = general.get<TimeUnitCommand>();
            m_Commands.push_back(std::make_unique<TimeUnitCommand>(timeUnitCommand));
        }

        if (general.contains("OutputFile"))
        {
            auto outputCommand = general.get<OutputCommand>();
            m_Commands.push_back(std::make_unique<OutputCommand>(outputCommand));
        }
    }

    if (!input.contains("KernelSpecification"))
    {
        throw KttException("The provided tuning file does not contain kernel specification, so the tuning cannot be launched");
    }

    auto& kernelSpecification = input["KernelSpecification"];

    if (!kernelSpecification.contains("Language"))
    {
        throw KttException("The kernel specification does not contain the used kernel language (CUDA, OpenCL, etc.)");
    }

    auto createTunerCommand = kernelSpecification.get<CreateTunerCommand>();
    m_Commands.push_back(std::make_unique<CreateTunerCommand>(createTunerCommand));

    if (!kernelSpecification.contains("KernelName") || !kernelSpecification.contains("KernelFile"))
    {
        throw KttException("The kernel specification does not contain proper kernel description (kernel name and file)");
    }

    auto addKernelCommand = kernelSpecification.get<AddKernelCommand>();
    m_Commands.push_back(std::make_unique<AddKernelCommand>(addKernelCommand));

    if (kernelSpecification.contains("GlobalSizeType"))
    {
        auto sizeTypeCommand = kernelSpecification.get<SizeTypeCommand>();
        m_Commands.push_back(std::make_unique<SizeTypeCommand>(sizeTypeCommand));
    }

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

    if (kernelSpecification.contains("CompilerOptions"))
    {
        auto compilerOptionsCommand = kernelSpecification.get<CompilerOptionsCommand>();
        m_Commands.push_back(std::make_unique<CompilerOptionsCommand>(compilerOptionsCommand));
    }

    if (kernelSpecification.contains("Arguments"))
    {
        auto argumentCommands = kernelSpecification["Arguments"].get<std::vector<AddArgumentCommand>>();

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

    const auto tuneCommand = kernelSpecification.get<TuneCommand>();
    m_Commands.push_back(std::make_unique<TuneCommand>(tuneCommand));
}

bool TuningLoader::ValidateFormat(const std::string& tuningScript)
{
    json_validator valitor;

    try
    {
        valitor.set_root_schema(TuningSchema);
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Validation of schema failed: " << exception.what() << std::endl;
        return false;
    }

    json input = json::parse(tuningScript);

    try
    {
        valitor.validate(input);
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Validation of the provided JSON tuning file failed: " << exception.what();
        return false;
    }

    return true;
}

} // namespace ktt
