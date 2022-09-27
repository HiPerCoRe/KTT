#include <Kernel/KernelConstraint/KernelConstraint.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<const KernelParameter*>& parameters) :
    m_Parameters(parameters)
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

} // namespace ktt
