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
    m_ExploredConfigurations(0)
{
    InitializeConfigurations();
}

ConfigurationData::~ConfigurationData()
{
    m_Searcher.Reset();
}

bool ConfigurationData::CalculateNextConfiguration(const KernelResult& previousResult)
{
    ++m_ExploredConfigurations;
    UpdateBestConfiguration(previousResult);

    if (!IsProcessed())
    {
        m_Searcher.CalculateNextConfiguration(previousResult);
        return true;
    }

    return false;
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

uint64_t ConfigurationData::GetRandomConfigurationIndex(const std::set<uint64_t>& excludedIndices) const
{
    if (excludedIndices.size() >= GetTotalConfigurationsCount())
    {
        throw KttException("Excluded indices must not contain all valid configuration indices during random configuration generation");
    }

    const uint64_t index = m_Generator.Generate(0, GetTotalConfigurationsCount() - 1, excludedIndices);
    return index;
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
    return static_cast<uint64_t>(m_ExploredConfigurations);
}

bool ConfigurationData::IsProcessed() const
{
    return GetExploredConfigurationsCount() >= GetTotalConfigurationsCount();
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
