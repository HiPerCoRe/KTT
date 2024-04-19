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

    if (previousResult.IsValid())
    {
        UpdateBestConfiguration(previousResult);
    }

    if (IsProcessed())
    {
        return false;
    }

    try
    {
        m_SearcherActive = m_Searcher.CalculateNextConfiguration(previousResult);
        const auto& currentConfiguration = GetCurrentConfiguration();
        Logger::LogInfo("Searcher selected configuration " + std::to_string(GetIndexForConfiguration(currentConfiguration)) + ": " + currentConfiguration.GetString());
    }
    catch (const std::runtime_error& error)
    {
        Logger::LogError(error.what());
        m_SearcherActive = false;
    }

    if (!m_SearcherActive)
    {
        Logger::LogInfo("Searcher failed to calculate next configuration for kernel " + m_Kernel.GetName());
    }

    return m_SearcherActive;
}

void ConfigurationData::ListConfigurations() const
{
    Logger::LogInfo("Listing all configurations for kernel " + m_Kernel.GetName());

    for (uint64_t index = 0; index < GetTotalConfigurationsCount(); ++index)
    {
        KernelConfiguration configuration = GetConfigurationForIndex(index);
        Logger::LogInfo("Configuration " + std::to_string(index) + ": " + configuration.GetString());
    }
}

KernelConfiguration ConfigurationData::GetConfigurationForIndex(const uint64_t index) const
{
    if (index >= GetTotalConfigurationsCount())
    {
        throw KttException("Invalid configuration index");
    }

    const ConfigurationForest* localForest = nullptr;
    uint64_t localIndex = index;

    for (const auto& forest : m_Forests)
    {
        const uint64_t localCount = forest->GetConfigurationsCount();

        if (localCount <= localIndex)
        {
            localIndex -= localCount;
            continue;
        }

        localForest = forest.get();
        break;
    }

    auto result = localForest->GetConfiguration(localIndex);
    result.Merge(m_BestConfiguration.first);
    return result;
}

uint64_t ConfigurationData::GetIndexForConfiguration(const KernelConfiguration& configuration) const
{
    const ConfigurationForest& localForest = GetLocalForest(configuration);
    uint64_t result = 0;

    for (const auto& forest : m_Forests)
    {
        if (&localForest == forest.get())
        {
            break;
        }

        result += forest->GetConfigurationsCount();
    }

    result += localForest.GetLocalConfigurationIndex(configuration);
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
    std::vector<KernelConfiguration> configurations;

    m_Kernel.EnumerateNeighbourConfigurations(configuration, [this, &configurations, maxDifferences, maxNeighbours]
        (const auto& neighbour, const uint64_t differences)
    {
        if (configurations.size() >= maxNeighbours)
        {
            return false;
        }

        if (differences > maxDifferences)
        {
            return false;
        }

        const bool validNeighbour = std::all_of(m_Forests.cbegin(), m_Forests.cend(), [&neighbour](const auto& forest)
        {
            return forest->IsConfigurationValid(neighbour);
        });

        if (validNeighbour)
        {
            const uint64_t index = GetIndexForConfiguration(neighbour);

            if (!ContainsKey(m_ExploredConfigurations, index))
            {
                configurations.push_back(neighbour);
            }
        }

        return true;
    });

    for (auto& neighbour : configurations)
    {
        neighbour.Merge(m_BestConfiguration.first);
    }

    return configurations;
}

uint64_t ConfigurationData::GetTotalConfigurationsCount() const
{
    uint64_t result = 0;

    for (const auto& forest : m_Forests)
    {
        result += forest->GetConfigurationsCount();
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
    std::vector<std::vector<std::future<void>>> futures;

    for (const auto& group : groups)
    {
        m_Forests.push_back(std::make_unique<ConfigurationForest>());
        futures.push_back(m_Forests.back()->Build(group, pool));
    }

    for (const auto& groupFutures : futures)
    {
        for (const auto& future : groupFutures)
        {
            future.wait();
        }
    }

    timer.Stop();

    const auto& time = TimeConfiguration::GetInstance();
    const uint64_t elapsedTime = time.ConvertFromNanoseconds(timer.GetElapsedTime());
    Logger::LogInfo("Total count of " + std::to_string(GetTotalConfigurationsCount()) + " configurations was generated in "
        + std::to_string(elapsedTime) + time.GetUnitTag());

    KernelConfiguration initialBest;

    for (const auto& forest : m_Forests)
    {
        initialBest.Merge(forest->GetConfiguration(0));
    }

    m_BestConfiguration = {initialBest, InvalidDuration};
    m_SearcherActive = true;
    m_Searcher.Initialize(*this);
    Logger::LogInfo("Searcher selected configuration " + std::to_string(GetIndexForConfiguration(m_Searcher.GetCurrentConfiguration())) + ": " + m_Searcher.GetCurrentConfiguration().GetString());
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

const ConfigurationForest& ConfigurationData::GetLocalForest(const KernelConfiguration& configuration) const
{
    KttAssert(!IsProcessed(), "This should not be called after configuration space exploration is finished.");
    const auto& pairs = configuration.GetPairs();

    if (pairs.empty())
    {
        return *m_Forests[0];
    }

    // Assume that local tree parameters are at the beginning. Merge operation pushes parameters from different trees to the end.
    // Each tree contains at least one parameter, so the first pair is guaranteed to belong to the local tree.
    const auto& localPair = pairs[0];

    for (const auto& forest : m_Forests)
    {
        if (forest->HasParameter(localPair.GetName()))
        {
            return *forest;
        }
    }

    KttError("Inconsistent tree or configuration data.");
    return *m_Forests[0];
}

} // namespace ktt
