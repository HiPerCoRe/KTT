#include <KttLoaderAssert.h>
#include <JsonCommandConverters.h>

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

void from_json(const json& j, AddArgumentCommand& command)
{
    ArgumentMemoryType memoryType;
    j.at("MemoryType").get_to(memoryType);

    ArgumentDataType type;
    j.at("Type").get_to(type);

    size_t size = 0;

    if (j.contains("Size"))
    {
        j.at("Size").get_to(size);
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

    float fillValue;
    j.at("FillValue").get_to(fillValue);

    command = AddArgumentCommand(memoryType, type, size, accessType, fillType, fillValue);
}

void from_json(const json& j, AddKernelCommand& command)
{
    std::string name;
    j.at("KernelName").get_to(name);

    std::string file;
    j.at("KernelFile").get_to(file);

    command = AddKernelCommand(name, file);
}

void from_json(const json& j, CompilerOptionsCommand& command)
{
    std::vector<std::string> options;
    j.at("CompilerOptions").get_to(options);

    command = CompilerOptionsCommand(options);
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
    ComputeApi api;
    j.at("Language").get_to(api);
    command = CreateTunerCommand(api);
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

void from_json(const json& j, SharedMemoryCommand& command)
{
    size_t memorySize = 0;
    j.at("SharedMemory").get_to(memorySize);

    command = SharedMemoryCommand(memorySize);
}

void from_json(const json& j, TimeUnitCommand& command)
{
    TimeUnit unit;
    j.at("TimeUnit").get_to(unit);

    command = TimeUnitCommand(unit);
}

} // namespace ktt
