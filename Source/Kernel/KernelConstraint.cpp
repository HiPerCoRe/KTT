#include <Kernel/KernelConstraint.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function) :
    m_Parameters(parameters),
    m_Function(function),
    m_GenericFunction(nullptr)
{
    for (const auto* parameter : parameters)
    {
        m_ParameterNames.push_back(parameter->GetName());
    }

    m_ValuesCache.reserve(m_Parameters.size());
}

KernelConstraint::KernelConstraint(const std::vector<const KernelParameter*>& parameters, GenericConstraintFunction function) :
    m_Parameters(parameters),
    m_Function(nullptr),
    m_GenericFunction(function)
{
    for (const auto* parameter : parameters)
    {
        m_ParameterNames.push_back(parameter->GetName());
    }
}

const std::vector<const KernelParameter*>& KernelConstraint::GetParameters() const
{
    return m_Parameters;
}

bool KernelConstraint::AffectsParameter(const std::string& name) const
{
    return ContainsElementIf(m_ParameterNames, [&name](const auto& parameterName)
    {
        return parameterName == name;
    });
}

bool KernelConstraint::HasAllParameters(const std::set<std::string>& parameterNames) const
{
    return GetAffectedParameterCount(parameterNames) == m_Parameters.size();
}

uint64_t KernelConstraint::GetAffectedParameterCount(const std::set<std::string>& parameterNames) const
{
    uint64_t count = 0;

    for (const auto& parameter : m_ParameterNames)
    {
        const bool parameterPresent = ContainsKey(parameterNames, parameter);

        if (parameterPresent)
        {
            ++count;
        }
    }

    return count;
}

bool KernelConstraint::IsFulfilled(const std::vector<const ParameterValue*>& values) const
{
    if (m_GenericFunction != nullptr)
    {
        return m_GenericFunction(values);
    }

    m_ValuesCache.clear();

    for (const auto& value : values)
    {
        m_ValuesCache.push_back(std::get<uint64_t>(*value));
    }

    return m_Function(m_ValuesCache);
}

} // namespace ktt
