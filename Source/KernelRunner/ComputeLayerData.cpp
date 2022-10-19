#include <string>

#include <Api/KttException.h>
#include <KernelRunner/ComputeLayerData.h>
#include <Utility/StlHelpers.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

ComputeLayerData::ComputeLayerData(const Kernel& kernel, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
    const KernelRunMode runMode) :
    m_Kernel(kernel),
    m_Configuration(configuration),
    m_RunMode(runMode),
    m_DataOverhead(0)
{
    for (const auto* definition : kernel.GetDefinitions())
    {
        const auto id = definition->GetId();
        m_ComputeData.insert({id, KernelComputeData(kernel, *definition, configuration, dimensions)});
    }
}

void ComputeLayerData::IncreaseOverhead(const Nanoseconds overhead)
{
    m_DataOverhead += overhead;
}

void ComputeLayerData::AddPartialResult(const ComputationResult& result)
{
    m_PartialResults.push_back(result);
}

void ComputeLayerData::AddArgumentOverride(const ArgumentId& id, const KernelArgument& argument)
{
    std::map<KernelDefinitionId, size_t> argumentIndices;

    for (const auto& data : m_ComputeData)
    {
        const size_t index = data.second.GetArgumentIndex(id);

        if (index != std::numeric_limits<size_t>::max())
        {
            argumentIndices[data.first] = index;
        }
    }

    m_ArgumentOverrides.erase(id);
    auto pair = m_ArgumentOverrides.insert({id, argument});

    for (const auto& index : argumentIndices)
    {
        m_ComputeData.find(index.first)->second.UpdateArgumentAtIndex(index.second, pair.first->second);
    }
}

void ComputeLayerData::SwapArguments(const KernelDefinitionId id, const ArgumentId& first, const ArgumentId& second)
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel " + m_Kernel.GetName() + " has no data for definition with id "
            + std::to_string(id));
    }

    m_ComputeData.find(id)->second.SwapArguments(first, second);
}

void ComputeLayerData::ChangeArguments(const KernelDefinitionId id, std::vector<KernelArgument*>& arguments)
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel " + m_Kernel.GetName() + " has no data for definition with id "
            + std::to_string(id));
    }

    for (size_t i = 0; i < arguments.size(); ++i)
    {
        const auto& argumentId = arguments[i]->GetId();

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

KernelRunMode ComputeLayerData::GetRunMode() const
{
    return m_RunMode;
}

const KernelComputeData& ComputeLayerData::GetComputeData(const KernelDefinitionId id) const
{
    if (!ContainsKey(m_ComputeData, id))
    {
        throw KttException("Kernel " + m_Kernel.GetName() + " has no data for definition with id "
            + std::to_string(id));
    }

    return m_ComputeData.find(id)->second;
}

KernelResult ComputeLayerData::GenerateResult(const Nanoseconds launcherDuration) const
{
    KernelResult result(m_Kernel.GetName(), m_Configuration, m_PartialResults);
    const Nanoseconds launcherOverhead = CalculateLauncherOverhead();
    KttAssert(launcherDuration >= launcherOverhead, "Launcher overhead must be lower than its total duration");
    const Nanoseconds actualLauncherDuration = launcherDuration - launcherOverhead;
    
    if (m_Kernel.HasLauncher())
    {
        result.SetExtraDuration(actualLauncherDuration);
        result.SetDataMovementOverhead(m_DataOverhead);
    }
    else
    {
        // For simple kernels without user launcher, total duration is the same as kernel duration, everything else is overhead.
        result.SetDataMovementOverhead(actualLauncherDuration + m_DataOverhead);
    }

    return result;
}

Nanoseconds ComputeLayerData::CalculateLauncherOverhead() const
{
    Nanoseconds result = m_DataOverhead;

    for (const auto& partialResult : m_PartialResults)
    {
        result += partialResult.GetDuration();
        result += partialResult.GetOverhead();
    }

    return result;
}

} // namespace ktt
