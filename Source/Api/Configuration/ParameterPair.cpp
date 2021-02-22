#include <Api/Configuration/ParameterPair.h>
#include <Api/KttException.h>
#include <Kernel/KernelParameter.h>
#include <Utility/NumericalUtilities.h>

namespace ktt
{

ParameterPair::ParameterPair(const std::string& name, const uint64_t value) :
    m_Name(&name),
    m_Value(value)
{}

ParameterPair::ParameterPair(const std::string& name, const double value) :
    m_Name(&name),
    m_Value(value)
{}

void ParameterPair::SetValue(const uint64_t value)
{
    m_Value = value;
}

void ParameterPair::SetValue(const double value)
{
    m_Value = value;
}

const std::string& ParameterPair::GetName() const
{
    return *m_Name;
}

std::string ParameterPair::GetString() const
{
    std::string result(GetName() + " " + GetValueString());
    return result;
}

std::string ParameterPair::GetValueString() const
{
    if (HasValueDouble())
    {
        return std::to_string(GetValueDouble());
    }

    return std::to_string(GetValue());
}

uint64_t ParameterPair::GetValue() const
{
    if (HasValueDouble())
    {
        throw KttException("Attempting to retrieve integer value from floating-point parameter pair");
    }

    return std::get<uint64_t>(m_Value);
}

double ParameterPair::GetValueDouble() const
{
    if (!HasValueDouble())
    {
        throw KttException("Attempting to retrieve floating-point value from integer parameter pair");
    }

    return std::get<double>(m_Value);
}

bool ParameterPair::HasValueDouble() const
{
    return std::holds_alternative<double>(m_Value);
}

bool ParameterPair::HasSameValue(const ParameterPair& other) const
{
    if (HasValueDouble() && !other.HasValueDouble())
    {
        return false;
    }

    if (HasValueDouble())
    {
        return FloatEquals(GetValueDouble(), other.GetValueDouble());
    }

    return GetValue() == other.GetValue();
}

} // namespace ktt
