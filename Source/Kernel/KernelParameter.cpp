#include <Kernel/KernelParameter.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelParameter::KernelParameter(const std::string& name, const std::vector<uint64_t>& values, const std::string& group) :
    m_Name(name),
    m_Group(group),
    m_Values(values)
{
    if (values.empty())
    {
        throw KttException("Kernel parameter must have at least one value defined");
    }
}

KernelParameter::KernelParameter(const std::string& name, const std::vector<double>& values, const std::string& group) :
    m_Name(name),
    m_Group(group),
    m_Values(values)
{
    if (values.empty())
    {
        throw KttException("Kernel parameter must have at least one value defined");
    }
}

const std::string& KernelParameter::GetName() const
{
    return m_Name;
}

const std::string& KernelParameter::GetGroup() const
{
    return m_Group;
}

size_t KernelParameter::GetValuesCount() const
{
    if (HasValuesDouble())
    {
        return GetValuesDouble().size();
    }

    return GetValues().size();
}

const std::vector<uint64_t>& KernelParameter::GetValues() const
{
    if (HasValuesDouble())
    {
        throw KttException("Attempting to retrieve integer values from floating-point kernel parameter");
    }

    return std::get<std::vector<uint64_t>>(m_Values);
}

const std::vector<double>& KernelParameter::GetValuesDouble() const
{
    if (!HasValuesDouble())
    {
        throw KttException("Attempting to retrieve floating-point value from integer parameter pair");
    }

    return std::get<std::vector<double>>(m_Values);
}

bool KernelParameter::HasValuesDouble() const
{
    return std::holds_alternative<std::vector<double>>(m_Values);
}

ParameterPair KernelParameter::GeneratePair(const size_t valueIndex) const
{
    if (valueIndex >= GetValuesCount())
    {
        throw KttException("Parameter value index is out of range");
    }

    if (HasValuesDouble())
    {
        const double value = GetValuesDouble()[valueIndex];
        return ParameterPair(*this, value);
    }

    const uint64_t value = GetValues()[valueIndex];
    return ParameterPair(*this, value);
}

std::vector<ParameterPair> KernelParameter::GeneratePairs() const
{
    std::vector<ParameterPair> result;

    for (size_t i = 0; i < GetValuesCount(); ++i)
    {
        result.push_back(GeneratePair(i));
    }

    return result;
}

bool KernelParameter::operator==(const KernelParameter& other) const
{
    return m_Name == other.m_Name;
}

bool KernelParameter::operator!=(const KernelParameter& other) const
{
    return !(*this == other);
}

bool KernelParameter::operator<(const KernelParameter& other) const
{
    return m_Name < other.m_Name;
}

} // namespace ktt
