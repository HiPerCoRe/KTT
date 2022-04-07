#include <TuningLoader/JsonCommandConverters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

void from_json(const json& j, AddKernelCommand& command)
{
    std::string name;
    j.at("KernelName").get_to(name);

    std::string source;
    j.at("KernelCode").get_to(source);

    command = AddKernelCommand(name, source);
}

void from_json(const json& j, CreateTunerCommand& command)
{
    ComputeApi api;
    j.at("Language").get_to(api);
    command = CreateTunerCommand(api);
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
            KttError("Unhandled parameter value type");
        }

        command = ParameterCommand(name, finalValues);
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
        KttError("Unhandled parameter value type");
    }

    command = ParameterCommand(name, finalValues);
}

} // namespace ktt
