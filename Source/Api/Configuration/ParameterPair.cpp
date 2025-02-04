#include <Api/Configuration/ParameterPair.h>
#include <Api/KttException.h>
#include <Kernel/KernelParameter.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/NumericalUtilities.h>

namespace ktt
{

ParameterPair::ParameterPair() :
    m_Value(static_cast<uint64_t>(0))
{}

ParameterPair::ParameterPair(const std::string& name, const ParameterValue& value) :
    m_Name(name),
    m_Value(value)
{}

void ParameterPair::SetValue(const ParameterValue& value)
{
    m_Value = value;
}

const std::string& ParameterPair::GetName() const
{
    return m_Name;
}

std::string ParameterPair::GetString() const
{
    std::string result(GetName() + " " + GetValueString());
    return result;
}

std::string ParameterPair::GetValueString() const
{
    switch (GetValueType())
    {
    case ParameterValueType::Int:
        return std::to_string(std::get<int64_t>(m_Value));
    case ParameterValueType::UnsignedInt:
        return std::to_string(std::get<uint64_t>(m_Value));
    case ParameterValueType::Double:
        return std::to_string(std::get<double>(m_Value));
    case ParameterValueType::Bool:
        return std::get<bool>(m_Value) ? "true" : "false";
    case ParameterValueType::String:
        return std::get<std::string>(m_Value);
    default:
        KttError("Unhandled parameter value type");
        return "";
    }
}

const ParameterValue& ParameterPair::GetValue() const
{
    return m_Value;
}

uint64_t ParameterPair::GetValueUint() const
{
    if (GetValueType() != ParameterValueType::UnsignedInt)
    {
        throw KttException("Attempting to retrieve unsigned integer value from parameter pair with different value type");
    }

    return std::get<uint64_t>(m_Value);
}

ParameterValueType ParameterPair::GetValueType() const
{
    return GetTypeFromValue(m_Value);
}

bool ParameterPair::HasSameValue(const ParameterPair& other) const
{
    if (GetValueType() != other.GetValueType())
    {
        return false;
    }

    switch (GetValueType())
    {
    case ParameterValueType::Int:
        return std::get<int64_t>(m_Value) == std::get<int64_t>(other.GetValue());
    case ParameterValueType::UnsignedInt:
        return GetValueUint() == other.GetValueUint();
    case ParameterValueType::Double:
        return FloatEquals(std::get<double>(m_Value), std::get<double>(other.GetValue()));
    case ParameterValueType::Bool:
        return std::get<bool>(m_Value) == std::get<bool>(other.GetValue());
    case ParameterValueType::String:
        return GetValueString() == other.GetValueString();
    default:
        KttError("Unhandled parameter value type");
        return false;
    }
}

ParameterValueType ParameterPair::GetTypeFromValue(const ParameterValue& value)
{
    if (std::holds_alternative<int64_t>(value))
    {
        return ParameterValueType::Int;
    }
    else if (std::holds_alternative<uint64_t>(value))
    {
        return ParameterValueType::UnsignedInt;
    }
    else if (std::holds_alternative<double>(value))
    {
        return ParameterValueType::Double;
    }
    else if (std::holds_alternative<bool>(value))
    {
        return ParameterValueType::Bool;
    }
    else if (std::holds_alternative<std::string>(value))
    {
        return ParameterValueType::String;
    }

    KttError("Unhandled parameter value type");
    return ParameterValueType::Int;
}

} // namespace ktt
