#pragma once

#include <set>
#include <utility>
#include <vector>
#include <memory>
#include <optional>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Searcher/Searcher.h>
#include <Kernel/Kernel.h>
#include <TuningRunner/ConfigurationForest.h>
#include <Utility/RandomIntGenerator.h>
#include <KttTypes.h>
#include <Api/Configuration/RefinementParameters.h>
#include <Api/Configuration/TuningPhase.h>
#include <TuningRunner/ConfigurationRefinement.h>

namespace ktt
{

class ConfigurationData
{
public:
    explicit ConfigurationData(Searcher& searcher, const Kernel& kernel, const bool isSeparateOptionsGroup = false,
        const KernelConfiguration& baseConfiguration = KernelConfiguration(),
        const std::optional<RefinementParameters>& refinementParameters = std::nullopt);
    ~ConfigurationData();

    bool CalculateNextConfiguration(const KernelResult& previousResult);
    void ListConfigurations() const;

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

    bool IsExplorationFinished() const;
    TuningPhase GetPhase() const;
    void FinishExploration();
    void ResumeExploration();
    void RepeatRefinement();
    void CancelRefinement();
    void SetRefinementParameters(const RefinementParameters& parameters);

    bool IsSeparateOptionsGroup() const;

private:
    std::vector<std::unique_ptr<ConfigurationForest>> m_Forests;
    std::set<uint64_t> m_ExploredConfigurations;
    std::pair<KernelConfiguration, Nanoseconds> m_BestConfiguration;
    mutable RandomIntGenerator<uint64_t> m_Generator;
    Searcher& m_Searcher;
    const Kernel& m_Kernel;
    bool m_SearcherActive;
    bool m_IsSeparateOptionsGroup;
    KernelConfiguration m_BaseConfiguration;

    std::vector<std::pair<KernelConfiguration, Nanoseconds>> m_History;
    bool m_ExplorationFinishedByUser;
    std::optional<RefinementParameters> m_RefinementParameters;
    std::unique_ptr<ConfigurationRefinement> m_Refinement;
    KernelConfiguration m_RefinementWinner;

    void TransitionToRefinement();

    void InitializeConfigurations();
    void UpdateBestConfiguration(const KernelResult& previousResult);
    const ConfigurationForest& GetLocalForest(const KernelConfiguration& configuration) const;
};

} // namespace ktt
