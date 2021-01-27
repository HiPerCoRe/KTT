#include <Kernel/KernelParameter.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelParameter::KernelParameter(const std::string& name, const std::vector<uint64_t>& values) :
    m_Name(name),
    m_Values(values)
{
    if (values.empty())
    {
        throw KttException("Kernel parameter must have at least one value defined");
    }
}

KernelParameter::KernelParameter(const std::string& name, const std::vector<double>& values) :
    m_Name(name),
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

bool KernelParameter::operator==(const KernelParameter& other) const
{
    return m_Name == other.m_Name;
}

bool KernelParameter::operator!=(const KernelParameter& other) const
{
    return !(*this == other);
}

} // namespace ktt
