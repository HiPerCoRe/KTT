#include <Commands/ParameterCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

ParameterCommand::ParameterCommand(const std::string& name, const std::string& valueType, const std::string& valueScript) :
    m_Name(name),
    m_ValueScript(valueScript),
    m_ValueType(GetValueTypeFromString(valueType))
{}

void ParameterCommand::Execute(TunerContext& context)
{
    const KernelId id = context.GetKernelId();
    context.GetTuner().AddScriptParameter(id, m_Name, m_ValueType, m_ValueScript);
}

CommandPriority ParameterCommand::GetPriority() const
{
    return CommandPriority::ParameterDefinition;
}

ParameterValueType ParameterCommand::GetValueTypeFromString(const std::string& valueType)
{
    if (valueType == "int")
    {
        return ParameterValueType::Int;
    }
    else if (valueType == "uint")
    {
        return ParameterValueType::UnsignedInt;
    }
    else if (valueType == "float" || valueType == "double")
    {
        return ParameterValueType::Double;
    }
    else if (valueType == "bool")
    {
        return ParameterValueType::Bool;
    }
    else if (valueType == "string")
    {
        return ParameterValueType::String;
    }

    KttLoaderError("Unhandled parameter value type");
    return ParameterValueType::UnsignedInt;
}

} // namespace ktt
