#include <Api/KttException.h>
#include <Kernel/KernelParameter.h>

namespace ktt
{

static const std::string DefaultGroup = "KTTDefaultGroup";

KernelParameter::KernelParameter(const std::string& name, const std::vector<ParameterValue>& values, const std::string& group) :
    m_Name(name),
    m_Group(group),
    m_Values(values)
{
    if (values.empty())
    {
        throw KttException("Kernel parameter must have at least one value defined");
    }

    if (group.empty())
    {
        m_Group = DefaultGroup;
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
    return m_Values.size();
}

const std::vector<ParameterValue>& KernelParameter::GetValues() const
{
    return m_Values;
}

ParameterValueType KernelParameter::GetValueType() const
{
    return ParameterPair::GetTypeFromValue(m_Values[0]);
}

ParameterPair KernelParameter::GeneratePair(const size_t valueIndex) const
{
    if (valueIndex >= GetValuesCount())
    {
        throw KttException("Parameter value index is out of range");
    }

    return ParameterPair(m_Name, m_Values[valueIndex]);
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
