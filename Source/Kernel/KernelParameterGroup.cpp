#include <algorithm>

#include <Kernel/KernelParameterGroup.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelParameterGroup::KernelParameterGroup(const std::string& name, const std::vector<const KernelParameter*>& parameters,
    const std::vector<const KernelConstraint*>& constraints) :
    m_Name(name),
    m_Parameters(parameters),
    m_Constraints(constraints)
{
    KttAssert(!parameters.empty(), "Kernel parameter group must have at least one parameter");
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
    std::set<const KernelParameter*> remainingParameters(m_Parameters.cbegin(), m_Parameters.cend());
    std::set<const KernelConstraint*> remainingConstraints(m_Constraints.cbegin(), m_Constraints.cend());
    
    std::vector<KernelParameterGroup> result;
    size_t subgroupNumber = 0;

    while (!remainingConstraints.empty())
    {
        std::set<const KernelParameter*> currentParameters;
        std::set<const KernelConstraint*> currentConstraints;
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

const KernelConstraint& KernelParameterGroup::GetNextConstraintToProcess(const std::set<const KernelConstraint*> processedConstraints,
    const std::set<std::string>& processedParameters) const
{
    std::vector<const KernelConstraint*> constraintsToProcess;

    for (const auto* constraint : m_Constraints)
    {
        if (!ContainsKey(processedConstraints, constraint))
        {
            constraintsToProcess.push_back(constraint);
        }
    }

    KttAssert(!constraintsToProcess.empty(), "Retrieving next constraint to process when there are no unprocessed constraints left");

    for (const auto* constraint : constraintsToProcess)
    {
        // Constraints with all intersecting parameters are prioritized
        if (constraint->HasAllParameters(processedParameters))
        {
            return *constraint;
        }
    }

    return **constraintsToProcess.cbegin();
}

void KernelParameterGroup::EnumerateParameterIndices(const std::function<void(std::vector<size_t>&,
    const std::vector<const KernelParameter*>&)>& enumerator) const
{
    std::vector<size_t> initialIndices;
    const auto parameters = GetParametersInEnumerationOrder();
    const auto evaluationLevels = GetConstraintEvaluationLevels();
    ComputeIndices(0, initialIndices, parameters, evaluationLevels, enumerator);
}

void KernelParameterGroup::ComputeIndices(const size_t currentIndex, std::vector<size_t>& indices,
    const std::vector<const KernelParameter*>& parameters,
    const std::map<size_t, std::vector<const KernelConstraint*>>& evaluationLevels,
    const std::function<void(std::vector<size_t>&, const std::vector<const KernelParameter*>&)>& enumerator) const
{
    if (currentIndex >= parameters.size())
    {
        enumerator(indices, parameters);
        return;
    }

    const auto& parameterValues = parameters[currentIndex]->GetValues();

    for (size_t i = 0; i < parameterValues.size(); ++i)
    {
        std::vector<size_t> newIndices = indices;
        newIndices.push_back(i);
        bool constraintsFulfilled = true;

        for (const auto* constraint : evaluationLevels.find(currentIndex)->second)
        {
            std::vector<uint64_t> values;

            for (const auto* parameter : constraint->GetParameters())
            {
                for (size_t index = 0; index < newIndices.size(); ++index)
                {
                    if (parameter == parameters[index])
                    {
                        values.push_back(parameter->GetValues()[newIndices[index]]);
                        break;
                    }
                }
            }

            constraintsFulfilled &= constraint->IsFulfilled(values);
        }

        if (constraintsFulfilled)
        {
            ComputeIndices(currentIndex + 1, newIndices, parameters, evaluationLevels, enumerator);
        }
    }
}

std::vector<const KernelParameter*> KernelParameterGroup::GetParametersInEnumerationOrder() const
{
    std::vector<const KernelParameter*> result;
    auto sortedConstraints = m_Constraints;

    std::sort(sortedConstraints.begin(), sortedConstraints.end(), [](const auto* left, const auto* right)
    {
        return left->GetParameters().size() < right->GetParameters().size();
    });

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
