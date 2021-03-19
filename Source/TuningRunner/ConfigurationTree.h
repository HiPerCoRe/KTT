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

    uint64_t GetDepth() const;
    uint64_t GetConfigurationCount() const;
    KernelConfiguration GetConfiguration(const uint64_t index) const;

private:
    std::map<const KernelParameter*, uint64_t> m_ParameterToLevel;
    std::unique_ptr<ConfigurationNode> m_Root;
    bool m_IsBuilt;

    void AddPaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters,
        std::set<std::string>& lockedParameters);
    void PrunePaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters);
    std::vector<size_t> PreprocessIndices(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters,
        std::vector<uint64_t>& levels);
    std::vector<uint64_t> GetLockedLevels(const std::set<std::string>& lockedParameters);
    static std::vector<std::pair<const KernelParameter*, size_t>> MergeParametersWithIndices(
        const std::vector<const KernelParameter*>& parameters, const std::vector<size_t>& indices);
};

} // namespace ktt
