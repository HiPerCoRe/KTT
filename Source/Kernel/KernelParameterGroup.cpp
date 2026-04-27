#include <algorithm>

#include <Kernel/KernelParameterGroup.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

namespace
{

struct ParameterPointerComparator
{
    bool operator()(const KernelParameter* a, const KernelParameter* b) const
    {
        return a->GetName() < b->GetName();
    }
};

struct ConstraintPointerComparator
{
    bool operator()(const KernelConstraint* a, const KernelConstraint* b) const
    {
        if (a == b)
        {
            return false;
        }
        // First compare by number of parameters
        const size_t aSize = a->GetParameters().size();
        const size_t bSize = b->GetParameters().size();
        if (aSize != bSize)
        {
            return aSize < bSize;
        }
        // Then compare by parameter names (sorted) to get deterministic ordering
        // Since the number of parameters is small (typically <= 3), we can compute sorted names on the fly
        auto getSortedNames = [](const KernelConstraint* c) -> std::vector<std::string>
        {
            std::vector<std::string> names;
            names.reserve(c->GetParameters().size());
            for (const auto* param : c->GetParameters())
            {
                names.push_back(param->GetName());
            }
            std::sort(names.begin(), names.end());
            return names;
        };
        const auto aNames = getSortedNames(a);
        const auto bNames = getSortedNames(b);
        if (aNames != bNames)
        {
            return aNames < bNames;
        }
        // Distinct constraints can share the same parameter signature (e.g., two
        // separately-added shared-memory limits over the same parameter set). Without
        // a pointer tiebreaker, std::set treats them as equivalent and silently drops
        // all but one, so the dropped constraints never restrict the tuning space.
        return a < b;
    }
};

} // anonymous namespace

KernelParameterGroup::KernelParameterGroup(const std::string& name, const std::vector<const KernelParameter*>& parameters,
    const std::vector<const KernelConstraint*>& constraints) :
    m_Name(name),
    m_Parameters(parameters),
    m_Constraints(constraints)
{
    KttAssert(!parameters.empty(), "Kernel parameter group must have at least one parameter");
    m_ValuesCache.reserve(m_Parameters.size());
}

const std::string& KernelParameterGroup::GetName() const
{
    return m_Name;
}

const std::vector<const KernelParameter*>& KernelParameterGroup::GetParameters() const
{
    return m_Parameters;
}

const std::vector<const KernelConstraint*>& KernelParameterGroup::GetConstraints() const
{
    return m_Constraints;
}

bool KernelParameterGroup::ContainsParameter(const KernelParameter& parameter) const
{
    return ContainsElement(m_Parameters, &parameter);
}

bool KernelParameterGroup::ContainsParameter(const std::string& parameter) const
{
    return ContainsElementIf(m_Parameters, [&parameter](const auto* currentParameter)
    {
        return currentParameter->GetName() == parameter;
    });
}

std::vector<KernelParameterGroup> KernelParameterGroup::GenerateSubgroups() const
{
    std::set<const KernelParameter*, ParameterPointerComparator> remainingParameters(m_Parameters.cbegin(), m_Parameters.cend());
    std::set<const KernelConstraint*, ConstraintPointerComparator> remainingConstraints(m_Constraints.cbegin(), m_Constraints.cend());
    
    std::vector<KernelParameterGroup> result;
    size_t subgroupNumber = 0;

    while (!remainingConstraints.empty())
    {
        std::set<const KernelParameter*, ParameterPointerComparator> currentParameters;
        std::set<const KernelConstraint*, ConstraintPointerComparator> currentConstraints;
        bool newAdded = true;

        while (newAdded)
        {
            newAdded = false;

            for (const auto* constraint : remainingConstraints)
            {
                const bool affectsCurrentParameter = std::any_of(currentParameters.cbegin(), currentParameters.cend(),
                    [constraint](const auto* parameter)
                {
                    return constraint->AffectsParameter(parameter->GetName());
                });

                if (!affectsCurrentParameter && !currentConstraints.empty())
                {
                    continue;
                }

                currentConstraints.insert(constraint);
                newAdded = true;

                for (const auto* constraintParameter : constraint->GetParameters())
                {
                    currentParameters.insert(constraintParameter);
                    remainingParameters.erase(constraintParameter);
                }
            }

            for (const auto* currentConstraint : currentConstraints)
            {
                remainingConstraints.erase(currentConstraint);
            }
        }

        result.emplace_back(m_Name + "_Subgroup" + std::to_string(subgroupNumber),
            std::vector<const KernelParameter*>(currentParameters.cbegin(), currentParameters.cend()),
            std::vector<const KernelConstraint*>(currentConstraints.cbegin(), currentConstraints.cend()));
        ++subgroupNumber;
    }

    for (const auto* parameter : remainingParameters)
    {
        result.emplace_back(m_Name + "_Subgroup" + std::to_string(subgroupNumber), std::vector<const KernelParameter*>{parameter},
            std::vector<const KernelConstraint*>{});
        ++subgroupNumber;
    }

    return result;
}

std::vector<const KernelParameter*> KernelParameterGroup::GetParametersInEnumerationOrder() const
{
    // Parameters which are part of constraints with low number of parameters are prioritized, so these constraints can be
    // evaluated sooner during configurations enumeration.
    std::vector<const KernelParameter*> result;
    auto sortedConstraints = m_Constraints;

    std::sort(sortedConstraints.begin(), sortedConstraints.end(), ConstraintPointerComparator());

    for (const auto* constraint : sortedConstraints)
    {
        for (const auto* parameter : constraint->GetParameters())
        {
            if (!ContainsElement(result, parameter))
            {
                result.push_back(parameter);
            }
        }
    }

    for (const auto* parameter : m_Parameters)
    {
        if (!ContainsElement(result, parameter))
        {
            result.push_back(parameter);
        }
    }

    return result;
}

void KernelParameterGroup::EnumerateParameterIndices(const std::function<void(const std::vector<size_t>&)>& enumerator) const
{
    std::vector<size_t> initialIndices;
    const auto evaluationLevels = GetConstraintEvaluationLevels();
    const auto parameters = GetParametersInEnumerationOrder();
    ComputeIndices(0, initialIndices, evaluationLevels, parameters, enumerator);
}

void KernelParameterGroup::ComputeIndices(const size_t currentIndex, const std::vector<size_t>& indices,
    const std::map<size_t, std::vector<const KernelConstraint*>>& evaluationLevels,
    const std::vector<const KernelParameter*>& parameters, const std::function<void(const std::vector<size_t>&)>& enumerator) const
{
    if (currentIndex >= parameters.size())
    {
        enumerator(indices);
        return;
    }

    for (size_t i = 0; i < parameters[currentIndex]->GetValuesCount(); ++i)
    {
        std::vector<size_t> newIndices = indices;
        newIndices.push_back(i);
        bool constraintsFulfilled = true;
        KttAssert(ContainsKey(evaluationLevels, currentIndex), "Invalid current index or evaluation levels");

        for (const auto* constraint : evaluationLevels.find(currentIndex)->second)
        {
            m_ValuesCache.clear();

            for (const auto* parameter : constraint->GetParameters())
            {
                for (size_t index = 0; index < newIndices.size(); ++index)
                {
                    if (parameter == parameters[index])
                    {
                        m_ValuesCache.push_back(&parameter->GetValues()[newIndices[index]]);
                        break;
                    }
                }
            }

            constraintsFulfilled &= constraint->IsFulfilled(m_ValuesCache);
        }

        if (constraintsFulfilled)
        {
            ComputeIndices(currentIndex + 1, newIndices, evaluationLevels, parameters, enumerator);
        }
    }
}

std::map<size_t, std::vector<const KernelConstraint*>> KernelParameterGroup::GetConstraintEvaluationLevels() const
{
    std::map<size_t, std::vector<const KernelConstraint*>> result;
    const auto orderedParameters = GetParametersInEnumerationOrder();
    std::set<std::string> includedParameters;
    std::set<const KernelConstraint*> processedConstraints;

    for (size_t level = 0; level < orderedParameters.size(); ++level)
    {
        result[level] = std::vector<const KernelConstraint*>{};
        includedParameters.insert(orderedParameters[level]->GetName());

        for (const auto* constraint : m_Constraints)
        {
            if (ContainsKey(processedConstraints, constraint))
            {
                continue;
            }

            if (constraint->HasAllParameters(includedParameters))
            {
                result[level].push_back(constraint);
                processedConstraints.insert(constraint);
            }
        }
    }

    return result;
}

} // namespace ktt
