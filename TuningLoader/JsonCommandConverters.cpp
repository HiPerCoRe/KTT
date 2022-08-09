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
    std::string options;
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

    ParameterValueType valueType;
    j.at("Type").get_to(valueType);

    std::vector<ParameterValue> finalValues;

    if (!j.contains("Values"))
    {
        const auto& range = j.at("Range");

        switch (valueType)
        {
        case ParameterValueType::Int:
        {
            int64_t lower;
            range.at("Lower").get_to(lower);

            int64_t upper;
            range.at("Upper").get_to(upper);

            int64_t step = 1;

            if (range.contains("Step"))
            {
                range.at("Step").get_to(step);
            }

            for (int64_t i = lower; i < upper; i += step)
            {
                finalValues.push_back(i);
            }

            break;
        }
        case ParameterValueType::UnsignedInt:
        {
            uint64_t lower;
            range.at("Lower").get_to(lower);

            uint64_t upper;
            range.at("Upper").get_to(upper);

            uint64_t step = 1;

            if (range.contains("Step"))
            {
                range.at("Step").get_to(step);
            }

            for (uint64_t i = lower; i < upper; i += step)
            {
                finalValues.push_back(i);
            }

            break;
        }
        case ParameterValueType::Double:
        {
            double lower;
            range.at("Lower").get_to(lower);

            double upper;
            range.at("Upper").get_to(upper);

            double step = 1;

            if (range.contains("Step"))
            {
                range.at("Step").get_to(step);
            }

            for (double i = lower; i < upper; i += step)
            {
                finalValues.push_back(i);
            }

            break;
        }
        default:
            KttLoaderError("Unhandled parameter value type");
        }

        command = ParameterCommand(valueType, name, finalValues);
        return;
    }

    switch (valueType)
    {
    case ParameterValueType::Int:
    {
        std::vector<int64_t> values;
        j.at("Values").get_to(values);

        for (const auto value : values)
        {
            finalValues.push_back(value);
        }

        break;
    }
    case ParameterValueType::UnsignedInt:
    {
        std::vector<uint64_t> values;
        j.at("Values").get_to(values);

        for (const auto value : values)
        {
            finalValues.push_back(value);
        }

        break;
    }
    case ParameterValueType::Double:
    {
        std::vector<double> values;
        j.at("Values").get_to(values);

        for (const auto value : values)
        {
            finalValues.push_back(value);
        }

        break;
    }
    case ParameterValueType::Bool:
    {
        std::vector<bool> values;
        j.at("Values").get_to(values);

        for (const auto value : values)
        {
            finalValues.push_back(value);
        }

        break;
    }
    case ParameterValueType::String:
    {
        std::vector<std::string> values;
        j.at("Values").get_to(values);

        for (const auto value : values)
        {
            finalValues.push_back(value);
        }

        break;
    }
    default:
        KttLoaderError("Unhandled parameter value type");
    }

    command = ParameterCommand(valueType, name, finalValues);
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
