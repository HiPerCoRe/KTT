#include <Kernel/KernelParameterGroup.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelParameterGroup::KernelParameterGroup(const std::string& name, const std::vector<const KernelParameter*>& parameters) :
    m_Name(name),
    m_Parameters(parameters)
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

uint64_t KernelParameterGroup::GetConfigurationsCount() const
{
    uint64_t result = 1;

    for (const auto& parameter : m_Parameters)
    {
        result *= static_cast<uint64_t>(parameter->GetValuesCount());
    }

    return result;
}

} // namespace ktt
