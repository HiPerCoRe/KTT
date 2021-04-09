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

} // namespace ktt
