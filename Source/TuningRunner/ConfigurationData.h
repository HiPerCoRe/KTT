#pragma once

#include <utility>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Searcher/Searcher.h>
#include <Kernel/Kernel.h>
#include <TuningRunner/ConfigurationTree.h>
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

    uint64_t GetTotalConfigurationsCount() const;
    uint64_t GetExploredConfigurationsCount() const;
    bool IsProcessed() const;
    KernelConfiguration GetCurrentConfiguration() const;
    KernelConfiguration GetBestConfiguration() const;

private:
    std::vector<std::unique_ptr<ConfigurationTree>> m_Trees;
    std::pair<KernelConfiguration, Nanoseconds> m_BestConfiguration;
    Searcher& m_Searcher;
    const Kernel& m_Kernel;
    size_t m_ExploredConfigurations;

    void InitializeConfigurations();
    void UpdateBestConfiguration(const KernelResult& previousResult);

    // Legacy configuration computation
    void ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
        std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const;
    bool IsConfigurationValid(const std::vector<ParameterPair>& pairs) const;
};

} // namespace ktt
