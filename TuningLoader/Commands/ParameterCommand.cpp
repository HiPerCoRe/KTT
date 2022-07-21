#include <Commands/ParameterCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

ParameterCommand::ParameterCommand(const ParameterValueType valueType, const std::string& name, const std::vector<ParameterValue>& values) :
    m_ValueType(valueType),
    m_Name(name),
    m_Values(values)
{}

void ParameterCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();

    switch (m_ValueType)
    {
    case ParameterValueType::Int:
    {
        std::vector<int64_t> values;

        for (const auto& value : m_Values)
        {
            values.push_back(std::get<int64_t>(value));
        }

        context.GetTuner().AddParameter(id, m_Name, values);
        break;
    }
    case ParameterValueType::UnsignedInt:
    {
        std::vector<uint64_t> values;

        for (const auto& value : m_Values)
        {
            values.push_back(std::get<uint64_t>(value));
        }

        context.GetTuner().AddParameter(id, m_Name, values);
        break;
    }
    case ParameterValueType::Double:
    {
        std::vector<double> values;

        for (const auto& value : m_Values)
        {
            values.push_back(std::get<double>(value));
        }

        context.GetTuner().AddParameter(id, m_Name, values);
        break;
    }
    case ParameterValueType::Bool:
    {
        std::vector<bool> values;

        for (const auto& value : m_Values)
        {
            values.push_back(std::get<bool>(value));
        }

        context.GetTuner().AddParameter(id, m_Name, values);
        break;
    }
    case ParameterValueType::String:
    {
        std::vector<std::string> values;

        for (const auto& value : m_Values)
        {
            values.push_back(std::get<std::string>(value));
        }

        context.GetTuner().AddParameter(id, m_Name, values);
        break;
    }
    default:
        KttLoaderError("Unhandled parameter value type");
    }
}

CommandPriority ParameterCommand::GetPriority() const
{
    return CommandPriority::ParameterDefinition;
}

} // namespace ktt
