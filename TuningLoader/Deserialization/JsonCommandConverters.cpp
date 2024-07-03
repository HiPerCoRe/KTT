#include <Deserialization/JsonCommandConverters.h>
#include <KttLoaderAssert.h>

namespace ktt
{

void to_json(json& j, const DimensionVector& vector)
{
    j = json
    {
        {"X", vector.GetSizeX()},
        {"Y", vector.GetSizeY()},
        {"Z", vector.GetSizeZ()}
    };
}

void from_json(const json& j, DimensionVector& vector)
{
    vector.SetSizeX(j.at("X").get<size_t>());
    vector.SetSizeY(j.at("Y").get<size_t>());
    vector.SetSizeZ(j.at("Z").get<size_t>());
}

void from_json(const json& j, SearcherAttribute& attribute)
{
    const auto name = j.at("Name").get<std::string>();
    const auto value = j.at("Value").get<std::string>();
    attribute = SearcherAttribute(name, value);
}

void from_json(const json& j, AddArgumentCommand& command)
{
    ArgumentId id = InvalidArgumentId;

    if (j.contains("Name"))
    {
        j.at("Name").get_to(id);
    }

    ArgumentMemoryType memoryType = ArgumentMemoryType::Vector;

    if (j.contains("MemoryType"))
    {
        j.at("MemoryType").get_to(memoryType);
    }

    ArgumentDataType type = ArgumentDataType::Float;

    if (j.contains("Type"))
    {
        j.at("Type").get_to(type);
    }

    size_t size = 0;

    if (j.contains("Size"))
    {
        j.at("Size").get_to(size);
    }

    size_t typeSize = 0;

    if (j.contains("TypeSize"))
    {
        j.at("TypeSize").get_to(typeSize);
    }

    ArgumentAccessType accessType = ArgumentAccessType::ReadWrite;

    if (j.contains("AccessType"))
    {
        j.at("AccessType").get_to(accessType);
    }
    
    ArgumentFillType fillType = ArgumentFillType::Constant;

    if (j.contains("FillType"))
    {
        j.at("FillType").get_to(fillType);
    }

    if (fillType == ArgumentFillType::Constant || fillType == ArgumentFillType::Random)
    {
        float fillValue;
        j.at("FillValue").get_to(fillValue);

        command = AddArgumentCommand(id, memoryType, type, size, typeSize, accessType, fillType, fillValue);
    }
    else if (fillType == ArgumentFillType::BinaryRaw || fillType == ArgumentFillType::Generator)
    {
        std::string dataSource;
        j.at("DataSource").get_to(dataSource);

        command = AddArgumentCommand(id, memoryType, type, size, typeSize, accessType, fillType, dataSource);
    }
}

void from_json(const json& j, AddKernelCommand& command)
{
    std::string name;
    j.at("KernelName").get_to(name);

    std::string file;
    j.at("KernelFile").get_to(file);

    std::vector<std::string> typeNames;

    if (j.contains("KernelTypeNames"))
    {
        j.at("KernelTypeNames").get_to(typeNames);
    }

    command = AddKernelCommand(name, file, typeNames);
}

void from_json(const json& j, CompilerOptionsCommand& command)
{
    std::vector<std::string> options;
    j.at("CompilerOptions").get_to(options);

    command = CompilerOptionsCommand(options);
}

void from_json(const json& j, ProfilingCommand& command)
{
    bool profilingOn;
    j.at("Profiling").get_to(profilingOn);

    command = ProfilingCommand(profilingOn);
}

void from_json(const json& j, ConstraintCommand& command)
{
    std::vector<std::string> parameters;
    j.at("Parameters").get_to(parameters);

    std::string expression;
    j.at("Expression").get_to(expression);

    command = ConstraintCommand(parameters, expression);
}

void from_json(const json& j, CreateTunerCommand& command)
{
    PlatformIndex platformId = 0;
    DeviceIndex deviceId = 0;
    ComputeApi api;
    if (j.contains("Device"))
    {
      if (j.at("Device").contains("PlatformId"))
          j.at("Device").at("PlatformId").get_to(platformId);
      if (j.at("Device").contains("DeviceId"))
          j.at("Device").at("DeviceId").get_to(deviceId);
    }
    j.at("Language").get_to(api);
    command = CreateTunerCommand(platformId, deviceId, api);

}

void from_json(const json& j, LoggingLevelCommand& command)
{
    LoggingLevel level;
    j.at("LoggingLevel").get_to(level);
    command = LoggingLevelCommand(level);
}

void from_json(const json& j, ModifierCommand& command)
{
    std::map<ModifierDimension, std::string> scripts;

    if (j.contains("X"))
    {
        scripts[ModifierDimension::X] = j.at("X").get<std::string>();
    }

    if (j.contains("Y"))
    {
        scripts[ModifierDimension::Y] = j.at("Y").get<std::string>();
    }

    if (j.contains("Z"))
    {
        scripts[ModifierDimension::Z] = j.at("Z").get<std::string>();
    }

    command = ModifierCommand(ModifierType::Global, scripts);
}

void from_json(const json& j, OutputCommand& command)
{
    std::string file;
    j.at("OutputFile").get_to(file);

    OutputFormat format = OutputFormat::JSON;

    if (j.contains("OutputFormat"))
    {
        j.at("OutputFormat").get_to(format);
    }

    command = OutputCommand(file, format);
}

void from_json(const json& j, ParameterCommand& command)
{
    std::string name;
    j.at("Name").get_to(name);

    std::string valueType;
    j.at("Type").get_to(valueType);

    std::string valueScript;
    j.at("Values").get_to(valueScript);

    command = ParameterCommand(name, valueType, valueScript);
}

void from_json(const json& j, SearcherCommand& command)
{
    const auto type = j.at("Name").get<SearcherType>();
    std::map<std::string, std::string> attributes;

    if (j.contains("Attributes"))
    {
        const auto attributesArray = j.at("Attributes").get<std::vector<SearcherAttribute>>();

        for (const auto& attribute : attributesArray)
        {
            attributes.insert(attribute.GeneratePair());
        }
    }

    command = SearcherCommand(type, attributes);
}

void from_json(const json& j, SharedMemoryCommand& command)
{
    size_t memorySize = 0;
    j.at("SharedMemory").get_to(memorySize);

    command = SharedMemoryCommand(memorySize);
}

void from_json(const json& j, SizeTypeCommand& command)
{
    const auto type = j.at("GlobalSizeType").get<GlobalSizeType>();

    command = SizeTypeCommand(type);
}

void from_json(const json& j, StopConditionCommand& command)
{
    std::vector<StopConditionType> types;

    for (auto it = j.begin(); it != j.end(); ++it)
    {
        const auto type = it.value()["Type"].get<StopConditionType>();
        types.push_back(type);
    }

    std::vector<double> budgets;

    for (auto it = j.begin(); it != j.end(); ++it)
    {
        const auto value = it.value()["BudgetValue"].get<double>();
        budgets.push_back(value);
    }

    command = StopConditionCommand(types, budgets);
}

void from_json(const json& j, TimeUnitCommand& command)
{
    TimeUnit unit;
    j.at("TimeUnit").get_to(unit);

    command = TimeUnitCommand(unit);
}

void from_json(const json& j, TuneCommand& command)
{
    if (j.contains("SimulationInput"))
    {
        const auto simulationInput = j.at("SimulationInput").get<std::string>();
        command = TuneCommand(simulationInput);
    }
}

void from_json(const json& j, ValidationCommand& command)
{
    const auto target = j.at("TargetName").get<ArgumentId>();
    const auto reference = j.get<AddArgumentCommand>();

    command = ValidationCommand(target, reference);
}

} // namespace ktt
