#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Kernel/KernelParameterGroup.h>
#include <TuningRunner/ConfigurationNode.h>

namespace ktt
{

class ConfigurationTree
{
public:
    ConfigurationTree();

    void Build(const KernelParameterGroup& group);
    void Clear();

    bool IsBuilt() const;
    bool HasParameter(const std::string& name) const;
    uint64_t GetDepth() const;
    uint64_t GetConfigurationsCount() const;
    KernelConfiguration GetConfiguration(const uint64_t index) const;
    std::vector<KernelConfiguration> GetNeighbourConfigurations(const KernelConfiguration& configuration,
        const uint64_t maxDifferences, const size_t maxNeighbours, const std::set<uint64_t> exploredConfigurations) const;
    uint64_t GetLocalConfigurationIndex(const KernelConfiguration& configuration) const;

private:
    std::map<const KernelParameter*, uint64_t> m_ParameterToLevel;
    std::unique_ptr<ConfigurationNode> m_Root;
    bool m_IsBuilt;

    void AddPaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters,
        std::set<std::string>& lockedParameters);
    void PrunePaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters);
    std::vector<size_t> PreprocessIndices(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters,
        std::vector<uint64_t>& levels);
    std::vector<uint64_t> GetParameterLevels(const std::set<std::string>& parameters) const;
    KernelConfiguration GetConfigurationFromIndices(const std::vector<size_t>& indices) const;
    std::vector<size_t> GetIndicesFromConfiguration(const KernelConfiguration& configuration) const;

    static std::vector<std::pair<const KernelParameter*, size_t>> MergeParametersWithIndices(
        const std::vector<const KernelParameter*>& parameters, const std::vector<size_t>& indices);
};

} // namespace ktt
