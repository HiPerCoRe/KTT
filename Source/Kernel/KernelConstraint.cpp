#include <Kernel/KernelConstraint.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<std::string>& parameters, ConstraintFunction function) :
    m_Parameters(parameters),
    m_Function(function)
{}

const std::vector<std::string>& KernelConstraint::GetParameters() const
{
    return m_Parameters;
}

bool KernelConstraint::HasAllParameters(const std::vector<ParameterPair>& pairs) const
{
    for (const auto& parameter : m_Parameters)
    {
        const bool parameterPresent = ContainsElementIf(pairs, [&parameter](const auto& pair)
        {
            return pair.GetName() == parameter;
        });

        if (!parameterPresent)
        {
            return false;
        }
    }

    return true;
}

bool KernelConstraint::IsFulfilled(const std::vector<ParameterPair>& pairs) const
{
    std::vector<uint64_t> values = ParameterPair::GetParameterValues<uint64_t>(pairs, m_Parameters);
    return m_Function(values);
}

} // namespace ktt
