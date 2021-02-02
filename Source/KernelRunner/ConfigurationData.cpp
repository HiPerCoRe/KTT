#include <limits>

#include <KernelRunner/ConfigurationData.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationData::ConfigurationData(const DeviceInfo& deviceInfo, Searcher& searcher, const Kernel& kernel) :
    m_DeviceInfo(deviceInfo),
    m_Searcher(searcher),
    m_Kernel(kernel),
    m_CurrentGroup(std::numeric_limits<size_t>::max())
{
    m_Groups = kernel.GenerateParameterGroups();

    if (!IsProcessed())
    {
        InitializeNextGroup();
    }
}

ConfigurationData::~ConfigurationData()
{
    m_Searcher.Reset();
}

void ConfigurationData::CalculateNextConfiguration(const KernelResult& previousResult)
{
    m_ProcessedConfigurations.insert(std::make_pair(previousResult.GetTotalDuration(), GetCurrentConfiguration()));
    m_Searcher.CalculateNextConfiguration(previousResult);
}

uint64_t ConfigurationData::GetConfigurationCount() const
{
    return static_cast<uint64_t>(m_Configurations.size());
}

bool ConfigurationData::IsProcessed() const
{
    return m_CurrentGroup >= m_Groups.size();
}

const KernelParameterGroup& ConfigurationData::GetCurrentGroup() const
{
    KttAssert(!IsProcessed(), "Current group can only be retrieved for configuration data that is not processed");
    return m_Groups[m_CurrentGroup];
}

const KernelConfiguration& ConfigurationData::GetCurrentConfiguration() const
{
    if (m_Configurations.empty() || m_Searcher.GetUnexploredConfigurationCount() == 0)
    {
        static KernelConfiguration defaultConfiguration;
        return defaultConfiguration;
    }

    return m_Searcher.GetCurrentConfiguration();
}

const KernelConfiguration& ConfigurationData::GetBestConfiguration() const
{
    if (!m_ProcessedConfigurations.empty())
    {
        const auto& bestPair = *m_ProcessedConfigurations.cbegin();
        return bestPair.second;
    }

    return GetCurrentConfiguration();
}

void ConfigurationData::InitializeNextGroup()
{
    m_Searcher.Reset();
    m_Configurations.clear();
    ++m_CurrentGroup;
    const auto& group = GetCurrentGroup();

    std::vector<KernelConfiguration> configurations;
    ComputeConfigurations(group, 0, std::vector<ParameterPair>{}, m_Configurations);
    m_Searcher.Initialize(m_Configurations);
}

void ConfigurationData::ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
    const std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentIndex >= group.GetParameters().size())
    {
        // All parameters are now included in the configuration
        std::vector<ParameterPair> extraPairs = GetExtraParameterPairs(pairs);
        std::vector<ParameterPair> allPairs;
        allPairs.reserve(pairs.size() + extraPairs.size());
        allPairs.insert(allPairs.end(), pairs.cbegin(), pairs.cend());
        allPairs.insert(allPairs.end(), extraPairs.cbegin(), extraPairs.cend());

        KernelConfiguration configuration(allPairs);

        if (IsConfigurationValid(configuration))
        {
            finalResult.push_back(configuration);
        }

        return;
    }

    if (!EvaluateConstraints(pairs))
    {
        return;
    }

    const KernelParameter& parameter = *group.GetParameters()[currentIndex]; 

    for (const auto& pair : parameter.GeneratePairs())
    {
        // Recursively build tree of configurations for each parameter value
        std::vector<ParameterPair> newPairs = pairs;
        newPairs.push_back(pair);
        ComputeConfigurations(group, currentIndex + 1, newPairs, finalResult);
    }
}

std::vector<ParameterPair> ConfigurationData::GetExtraParameterPairs(const std::vector<ParameterPair>& pairs) const
{
    std::vector<std::string> addedParameters;
    const auto& currentGroup = GetCurrentGroup();

    for (const auto* parameter : currentGroup.GetParameters())
    {
        addedParameters.push_back(parameter->GetName());
    }

    std::vector<ParameterPair> result;
    KernelConfiguration bestConfiguration;
    const bool valid = GetBestCompatibleConfiguration(pairs, bestConfiguration);

    if (valid)
    {
        for (const auto& bestPair : bestConfiguration.GetPairs())
        {
            if (!ContainsElement(addedParameters, bestPair.GetName()))
            {
                result.push_back(bestPair);
                addedParameters.push_back(bestPair.GetName());
            }
        }
    }

    for (const auto& parameter : m_Kernel.GetParameters())
    {
        if (!ContainsElement(addedParameters, parameter.GetName()))
        {
            result.push_back(parameter.GeneratePair(0));
            addedParameters.push_back(parameter.GetName());
        }
    }

    return result;
}

bool ConfigurationData::GetBestCompatibleConfiguration(const std::vector<ParameterPair>& pairs, KernelConfiguration& output) const
{
    for (const auto& configuration : m_ProcessedConfigurations)
    {
        if (IsConfigurationCompatible(pairs, configuration.second))
        {
            output = configuration.second;
            return true;
        }
    }

    return false;
}

bool ConfigurationData::IsConfigurationCompatible(const std::vector<ParameterPair>& pairs,
    const KernelConfiguration& configuration) const
{
    for (const auto& pair : pairs)
    {
        for (const auto& configurationPair : configuration.GetPairs())
        {
            if (pair.GetName() == configurationPair.GetName())
            {
                if (!pair.HasSameValue(configurationPair))
                {
                    return false;
                }

                break;
            }
        }
    }

    return true;
}

bool ConfigurationData::IsConfigurationValid(const KernelConfiguration& configuration) const
{
    const std::vector<ParameterPair>& pairs = configuration.GetPairs();

    for (const auto* definition : m_Kernel.GetDefinitions())
    {
        DimensionVector localSize = m_Kernel.GetModifiedLocalSize(definition->GetId(), pairs);

        if (localSize.GetTotalSize() > static_cast<size_t>(m_DeviceInfo.GetMaxWorkGroupSize()))
        {
            return false;
        }
    }

    if (!EvaluateConstraints(pairs))
    {
        return false;
    }

    return true;
}

bool ConfigurationData::EvaluateConstraints(const std::vector<ParameterPair>& pairs) const
{
    for (const auto& constraint : m_Kernel.GetConstraints())
    {
        if (!constraint.IsFulfilled(pairs))
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt
