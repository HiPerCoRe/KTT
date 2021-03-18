#include <Kernel/KernelConstraint.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function) :
    m_Parameters(parameters),
    m_Function(function)
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

const std::vector<std::string>& KernelConstraint::GetParameterNames() const
{
    return m_ParameterNames;
}

bool KernelConstraint::AffectsParameter(const std::string& name) const
{
    return ContainsElementIf(m_ParameterNames, [&name](const auto& parameterName)
    {
        return parameterName == name;
    });
}

bool KernelConstraint::HasAllParameters(const std::vector<ParameterPair>& pairs) const
{
    for (const auto* parameter : m_Parameters)
    {
        const bool parameterPresent = ContainsElementIf(pairs, [parameter](const auto& pair)
        {
            return pair.GetName() == parameter->GetName();
        });

        if (!parameterPresent)
        {
            return false;
        }
    }

    return true;
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

bool KernelConstraint::IsFulfilled(const std::vector<ParameterPair>& pairs) const
{
    std::vector<uint64_t> values = ParameterPair::GetParameterValues<uint64_t>(pairs, m_ParameterNames);
    return m_Function(values);
}

void KernelConstraint::EnumeratePairs(const std::function<void(std::vector<ParameterPair>&, const bool)>& enumerator) const
{
    std::vector<ParameterPair> initialPairs;
    ComputePairs(0, initialPairs, enumerator);
}

void KernelConstraint::EnumerateParameterIndices(const std::function<void(std::vector<size_t>&, const bool)>& enumerator) const
{
    std::vector<size_t> initialIndices;
    std::vector<uint64_t> initialValues;
    ComputeIndices(0, initialIndices, initialValues, enumerator);
}

void KernelConstraint::ComputePairs(const size_t currentIndex, std::vector<ParameterPair>& pairs,
    const std::function<void(std::vector<ParameterPair>&, const bool)>& enumerator) const
{
    if (currentIndex >= m_Parameters.size())
    {
        const bool isFulfilled = IsFulfilled(pairs);
        enumerator(pairs, isFulfilled);
        return;
    }

    const KernelParameter& parameter = *m_Parameters[currentIndex];

    for (const auto& pair : parameter.GeneratePairs())
    {
        std::vector<ParameterPair> newPairs = pairs;
        newPairs.push_back(pair);
        ComputePairs(currentIndex + 1, newPairs, enumerator);
    }
}

void KernelConstraint::ComputeIndices(const size_t currentIndex, std::vector<size_t>& indices, const std::vector<uint64_t>& values,
    const std::function<void(std::vector<size_t>&, const bool)>& enumerator) const
{
    if (currentIndex >= m_Parameters.size())
    {
        const bool isFulfilled = m_Function(values);
        enumerator(indices, isFulfilled);
        return;
    }

    const auto& parameterValues = m_Parameters[currentIndex]->GetValues();

    for (size_t i = 0; i < parameterValues.size(); ++i)
    {
        std::vector<size_t> newIndices = indices;
        newIndices.push_back(i);

        std::vector<uint64_t> newValues = values;
        newValues.push_back(parameterValues[i]);

        ComputeIndices(currentIndex + 1, newIndices, newValues, enumerator);
    }
}

} // namespace ktt
