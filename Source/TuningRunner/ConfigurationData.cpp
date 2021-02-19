#include <algorithm>
#include <limits>

#include <TuningRunner/ConfigurationData.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationData::ConfigurationData(const DeviceInfo& deviceInfo, Searcher& searcher, const Kernel& kernel) :
    m_DeviceInfo(deviceInfo),
    m_Searcher(searcher),
    m_Kernel(kernel),
    m_CurrentGroup(0)
{
    m_Groups = kernel.GenerateParameterGroups();

    if (!IsProcessed())
    {
        InitializeNextGroup(true);
    }
}

ConfigurationData::~ConfigurationData()
{
    m_Searcher.Reset();
}

bool ConfigurationData::CalculateNextConfiguration(const KernelResult& previousResult)
{
    m_ProcessedConfigurations.insert(std::make_pair(previousResult.GetTotalDuration(), GetCurrentConfiguration()));
    m_Searcher.CalculateNextConfiguration(previousResult);

    if (m_Searcher.GetUnexploredConfigurationCount() > 0)
    {
        return true;
    }

    return InitializeNextGroup(false);
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

bool ConfigurationData::InitializeNextGroup(const bool isInitialGroup)
{
    m_Searcher.Reset();
    m_Configurations.clear();

    if (!isInitialGroup)
    {
        ++m_CurrentGroup;
    }

    if (IsProcessed())
    {
        return false;
    }

    const auto& group = GetCurrentGroup();
    std::vector<ParameterPair> initialPairs;
    ComputeConfigurations(group, 0, initialPairs, m_Configurations);
    m_Searcher.Initialize(m_Configurations);

    Logger::LogInfo("Starting to explore configurations for kernel " + std::to_string(m_Kernel.GetId()) + " and group "
        + group.GetName() + ", configuration count in the current group is " + std::to_string(GetConfigurationCount()));
    return true;
}

void ConfigurationData::ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
    std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentIndex >= group.GetParameters().size())
    {
        // All parameters are now included in the configuration
        AddExtraParameterPairs(pairs);

        if (IsConfigurationValid(pairs))
        {
            finalResult.emplace_back(pairs);
        }

        return;
    }

    const KernelParameter& parameter = *group.GetParameters()[currentIndex]; 

    for (const auto& pair : parameter.GeneratePairs())
    {
        // Recursively build tree of configurations for each parameter value
        std::vector<ParameterPair> newPairs = pairs;
        newPairs.push_back(pair);

        if (!EvaluateConstraints(newPairs))
        {
            continue;
        }

        ComputeConfigurations(group, currentIndex + 1, newPairs, finalResult);
    }
}

void ConfigurationData::AddExtraParameterPairs(std::vector<ParameterPair>& pairs) const
{
    KernelConfiguration bestConfiguration;
    const bool valid = GetBestCompatibleConfiguration(pairs, bestConfiguration);

    if (valid)
    {
        for (const auto& bestPair : bestConfiguration.GetPairs())
        {
            const bool hasPair = ContainsElementIf(pairs, [&bestPair](const auto& pair)
            {
                return pair.GetName() == bestPair.GetName();
            });

            if (!hasPair)
            {
                pairs.push_back(bestPair);
            }
        }

        return;
    }

    for (const auto& parameter : m_Kernel.GetParameters())
    {
        const bool hasPair = ContainsElementIf(pairs, [&parameter](const auto& pair)
        {
            return pair.GetName() == parameter.GetName();
        });

        if (!hasPair)
        {
            pairs.push_back(parameter.GeneratePair(0));
        }
    }
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
        const auto& configurationPairs = configuration.GetPairs();

        const auto iterator = std::find_if(configurationPairs.cbegin(), configurationPairs.cend(),
            [&pair](const auto& configurationPair)
        {
            return pair.GetName() == configurationPair.GetName();
        });

        if (iterator == configurationPairs.cend())
        {
            continue;
        }

        if (!pair.HasSameValue(*iterator))
        {
            return false;
        }
    }

    return true;
}

bool ConfigurationData::IsConfigurationValid(const std::vector<ParameterPair>& pairs) const
{
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
        if (!constraint.HasAllParameters(pairs))
        {
            continue;
        }

        if (!constraint.IsFulfilled(pairs))
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt
