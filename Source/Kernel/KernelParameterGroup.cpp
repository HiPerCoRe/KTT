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

uint64_t KernelParameterGroup::GetConfigurationsCount() const
{
    uint64_t result = 1;

    for (const auto& parameter : m_Parameters)
    {
        result *= static_cast<uint64_t>(parameter->GetValuesCount());
    }

    return result;
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
