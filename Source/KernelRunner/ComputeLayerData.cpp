#include <string>

#include <KernelRunner/ComputeLayerData.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ComputeLayerData::ComputeLayerData(const Kernel& kernel, const KernelConfiguration& configuration) :
    m_Kernel(kernel),
    m_Configuration(configuration),
    m_Duration(0),
    m_Overhead(0)
{
    for (const auto* definition : kernel.GetDefinitions())
    {
        const auto id = definition->GetId();
        m_ComputeData.insert({id, KernelComputeData(kernel, *definition, configuration)});
    }
}

void ComputeLayerData::IncreaseDuration(const Nanoseconds duration)
{
    m_Duration += duration;
}

void ComputeLayerData::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void ComputeLayerData::AddPartialResult(const ComputationResult& result)
{
    m_PartialResults.push_back(result);
}

void ComputeLayerData::AddArgumentOverride(const ArgumentId id, const KernelArgument& argument)
{
    m_ArgumentOverrides.insert({id, argument});
    auto& storedArgument = m_ArgumentOverrides.find(id)->second;

    for (auto& data : m_ComputeData)
    {
        data.second.UpdateArgument(id, storedArgument);
    }
}

void ComputeLayerData::SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second)
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel with id " + std::to_string(m_Kernel.GetId()) + " has no data for definition with id "
            + std::to_string(id));
    }

    m_ComputeData.find(id)->second.SwapArguments(first, second);
}

void ComputeLayerData::ChangeArguments(const KernelDefinitionId id, std::vector<KernelArgument*>& arguments)
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel with id " + std::to_string(m_Kernel.GetId()) + " has no data for definition with id "
            + std::to_string(id));
    }

    for (size_t i = 0; i < arguments.size(); ++i)
    {
        const auto argumentId = arguments[i]->GetId();

        if (ContainsKey(m_ArgumentOverrides, argumentId))
        {
            arguments[i] = &m_ArgumentOverrides.find(argumentId)->second;
        }
    }

    m_ComputeData.find(id)->second.SetArguments(arguments);
}

bool ComputeLayerData::IsProfilingEnabled(const KernelDefinitionId id) const
{
    return ContainsElementIf(m_Kernel.GetProfiledDefinitions(), [id](const auto* definition)
    {
        return definition->GetId() == id;
    });
}

const Kernel& ComputeLayerData::GetKernel() const
{
    return m_Kernel;
}

const KernelConfiguration& ComputeLayerData::GetConfiguration() const
{
    return m_Configuration;
}

const KernelComputeData& ComputeLayerData::GetComputeData(const KernelDefinitionId id) const
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel with id " + std::to_string(m_Kernel.GetId()) + " has no data for definition with id "
            + std::to_string(id));
    }

    return m_ComputeData.find(id)->second;
}

KernelResult ComputeLayerData::GenerateResult(const Nanoseconds launcherDuration) const
{
    KernelResult result(m_Kernel.GetId(), m_PartialResults);
    const Nanoseconds launcherWithoutOverhead = launcherDuration - m_Overhead;
    result.SetExtraDuration(m_Duration + launcherWithoutOverhead);
    result.SetExtraOverhead(m_Overhead);
    return result;
}

} // namespace ktt
