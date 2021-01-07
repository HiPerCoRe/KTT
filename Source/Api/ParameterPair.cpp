#include <Api/ParameterPair.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

ParameterPair::ParameterPair() :
    m_Name(""),
    m_Value(0ull)
{}

ParameterPair::ParameterPair(const std::string& name, const uint64_t value) :
    m_Name(name),
    m_Value(value)
{}

ParameterPair::ParameterPair(const std::string& name, const double value) :
    m_Name(name),
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
    return m_Name;
}

std::string ParameterPair::GetString() const
{
    std::string result(m_Name + ": ");

    if (HasValueDouble())
    {
        const double value = GetValueDouble();
        result += std::to_string(value);
    }
    else
    {
        const uint64_t value = GetValue();
        result += std::to_string(value);
    }

    return result;
}

uint64_t ParameterPair::GetValue() const
{
    KttAssert(!HasValueDouble(), "Attempting to retrieve integer value from floating-point parameter pair");
    return std::get<uint64_t>(m_Value);
}

double ParameterPair::GetValueDouble() const
{
    KttAssert(HasValueDouble(), "Attempting to retrieve floating-point value from integer parameter pair");
    return std::get<double>(m_Value);
}

bool ParameterPair::HasValueDouble() const
{
    return std::holds_alternative<double>(m_Value);
}

} // namespace ktt
