#include <algorithm>
#include <limits>
#include <ctpl_stl.h>

#include <Api/KttException.h>
#include <Kernel/KernelParameterGroup.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <TuningRunner/ConfigurationData.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

ConfigurationData::ConfigurationData(Searcher& searcher, const Kernel& kernel) :
    m_BestConfiguration({KernelConfiguration(), InvalidDuration}),
    m_Searcher(searcher),
    m_Kernel(kernel),
    m_SearcherActive(false)
{
    InitializeConfigurations();
}

ConfigurationData::~ConfigurationData()
{
    m_Searcher.Reset();
}

bool ConfigurationData::CalculateNextConfiguration(const KernelResult& previousResult)
{
    const auto& previousConfiguration = previousResult.GetConfiguration();
    const uint64_t index = GetIndexForConfiguration(previousConfiguration);
    m_ExploredConfigurations.insert(index);

    UpdateBestConfiguration(previousResult);

    if (IsProcessed())
    {
        return false;
    }

    m_SearcherActive = m_Searcher.CalculateNextConfiguration(previousResult);

    if (!m_SearcherActive)
    {
        Logger::LogInfo("Searcher failed to calculate next configuration for kernel " + m_Kernel.GetName());
    }

    return m_SearcherActive;
}

KernelConfiguration ConfigurationData::GetConfigurationForIndex(const uint64_t index) const
{
    if (index >= GetTotalConfigurationsCount())
    {
        throw KttException("Invalid configuration index");
    }

    const ConfigurationTree* localTree = nullptr;
    uint64_t localIndex = index;

    for (const auto& tree : m_Trees)
    {
        const uint64_t localCount = tree->GetConfigurationsCount();

        if (localCount <= localIndex)
        {
            localIndex -= localCount;
            continue;
        }

        localTree = tree.get();
        break;
    }

    auto result = localTree->GetConfiguration(localIndex);
    result.Merge(m_BestConfiguration.first);
    return result;
}

uint64_t ConfigurationData::GetIndexForConfiguration(const KernelConfiguration& configuration) const
{
    const ConfigurationTree& localTree = GetLocalTree(configuration);
    uint64_t result = 0;

    for (const auto& tree : m_Trees)
    {
        if (&localTree == tree.get())
        {
            break;
        }

        result += tree->GetConfigurationsCount();
    }

    result += localTree.GetLocalConfigurationIndex(configuration);
    --result;
    KttAssert(result < GetTotalConfigurationsCount(), "Invalid computed configuration index");
    return result;
}

KernelConfiguration ConfigurationData::GetRandomConfiguration() const
{
    KttAssert(!IsProcessed(), "This should not be called after configuration space exploration is finished.");
    const uint64_t index = m_Generator.Generate(0, GetTotalConfigurationsCount() - 1, m_ExploredConfigurations);
    return GetConfigurationForIndex(index);
}

std::vector<KernelConfiguration> ConfigurationData::GetNeighbourConfigurations(const KernelConfiguration& configuration,
    const uint64_t maxDifferences, const size_t maxNeighbours) const
{
    const ConfigurationTree& localTree = GetLocalTree(configuration);
    auto configurations = localTree.GetNeighbourConfigurations(configuration, maxDifferences, maxNeighbours,
        m_ExploredConfigurations);

    for (auto& neighbour : configurations)
    {
        neighbour.Merge(m_BestConfiguration.first);
    }

    return configurations;
}

uint64_t ConfigurationData::GetTotalConfigurationsCount() const
{
    uint64_t result = 0;

    for (const auto& tree : m_Trees)
    {
        result += tree->GetConfigurationsCount();
    }

    return result;
}

uint64_t ConfigurationData::GetExploredConfigurationsCount() const
{
    return static_cast<uint64_t>(m_ExploredConfigurations.size());
}

const std::set<uint64_t>& ConfigurationData::GetExploredConfigurations() const
{
    return m_ExploredConfigurations;
}

bool ConfigurationData::IsProcessed() const
{
    return GetExploredConfigurationsCount() >= GetTotalConfigurationsCount() || !m_SearcherActive;
}

KernelConfiguration ConfigurationData::GetCurrentConfiguration() const
{
    if (!m_Searcher.IsInitialized() || IsProcessed())
    {
        return KernelConfiguration();
    }

    return m_Searcher.GetCurrentConfiguration();
}

KernelConfiguration ConfigurationData::GetBestConfiguration() const
{
    if (m_BestConfiguration.second != InvalidDuration)
    {
        return m_BestConfiguration.first;
    }

    return GetCurrentConfiguration();
}

void ConfigurationData::InitializeConfigurations()
{
    const auto groups = m_Kernel.GenerateParameterGroups();
    Logger::LogInfo("Generating configurations for kernel " + m_Kernel.GetName());

    Timer timer;
    timer.Start();

    ctpl::thread_pool pool;

    for (size_t i = 0; i < groups.size(); ++i)
    {
        auto newTree = std::make_unique<ConfigurationTree>();
        m_Trees.push_back(std::move(newTree));

        auto& tree = *m_Trees[i].get();
        auto& group = groups[i];

        pool.push([&tree, &group]()
        {
            tree.Build(group);
        });
    }

    pool.wait();

    timer.Stop();

    const auto& time = TimeConfiguration::GetInstance();
    const uint64_t elapsedTime = time.ConvertFromNanoseconds(timer.GetElapsedTime());
    Logger::LogInfo("Configurations were generated in " + std::to_string(elapsedTime) + time.GetUnitTag());

    KernelConfiguration initialBest;

    for (const auto& tree : m_Trees)
    {
        initialBest.Merge(tree->GetConfiguration(0));
    }

    m_BestConfiguration = {initialBest, InvalidDuration};
    m_SearcherActive = true;
    m_Searcher.Initialize(*this);
}

void ConfigurationData::UpdateBestConfiguration(const KernelResult& previousResult)
{
    const auto& configuration = previousResult.GetConfiguration();
    const Nanoseconds duration = previousResult.GetTotalDuration();

    if (duration < m_BestConfiguration.second)
    {
        m_BestConfiguration.first = configuration;
        m_BestConfiguration.second = duration;
    }
}

const ConfigurationTree& ConfigurationData::GetLocalTree(const KernelConfiguration& configuration) const
{
    KttAssert(!IsProcessed(), "This should not be called after configuration space exploration is finished.");
    const auto& pairs = configuration.GetPairs();

    if (pairs.empty())
    {
        return *m_Trees[0];
    }

    // Assume that local tree parameters are at the beginning. Merge operation pushes parameters from different trees to the end.
    // Each tree contains at least one parameter, so the first pair is guaranteed to belong to the local tree.
    const auto& localPair = pairs[0];

    for (const auto& tree : m_Trees)
    {
        if (tree->HasParameter(localPair.GetName()))
        {
            return *tree;
        }
    }

    KttError("Inconsistent tree or configuration data.");
    return *m_Trees[0];
}

void ConfigurationData::ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
    std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentIndex >= group.GetParameters().size())
    {
        // All parameters are now included in the configuration
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

        if (!IsConfigurationValid(newPairs))
        {
            continue;
        }

        ComputeConfigurations(group, currentIndex + 1, newPairs, finalResult);
    }
}

bool ConfigurationData::IsConfigurationValid(const std::vector<ParameterPair>& pairs) const
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
