#pragma once

#include <set>
#include <utility>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Searcher/Searcher.h>
#include <Kernel/Kernel.h>
#include <TuningRunner/ConfigurationTree.h>
#include <Utility/RandomIntGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class ConfigurationData
{
public:
    explicit ConfigurationData(Searcher& searcher, const Kernel& kernel);
    ~ConfigurationData();

    bool CalculateNextConfiguration(const KernelResult& previousResult);

    KernelConfiguration GetConfigurationForIndex(const uint64_t index) const;
    uint64_t GetIndexForConfiguration(const KernelConfiguration& configuration) const;
    KernelConfiguration GetRandomConfiguration() const;
    std::vector<KernelConfiguration> GetNeighbourConfigurations(const KernelConfiguration& configuration,
        const uint64_t maxDifferences, const size_t maxNeighbours = 3) const;

    uint64_t GetTotalConfigurationsCount() const;
    uint64_t GetExploredConfigurationsCount() const;
    const std::set<uint64_t>& GetExploredConfigurations() const;
    bool IsProcessed() const;
    KernelConfiguration GetCurrentConfiguration() const;
    KernelConfiguration GetBestConfiguration() const;

private:
    std::vector<std::unique_ptr<ConfigurationTree>> m_Trees;
    std::set<uint64_t> m_ExploredConfigurations;
    std::pair<KernelConfiguration, Nanoseconds> m_BestConfiguration;
    mutable RandomIntGenerator<uint64_t> m_Generator;
    Searcher& m_Searcher;
    const Kernel& m_Kernel;
    bool m_SearcherActive;

    void InitializeConfigurations();
    void UpdateBestConfiguration(const KernelResult& previousResult);
    const ConfigurationTree& GetLocalTree(const KernelConfiguration& configuration) const;

    // Legacy configuration computation
    void ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
        std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const;
    bool IsConfigurationValid(const std::vector<ParameterPair>& pairs) const;
};

} // namespace ktt
