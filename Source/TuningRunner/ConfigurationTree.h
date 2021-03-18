#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
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

private:
    std::map<std::string, uint64_t> m_ParameterToLevel;
    std::unique_ptr<ConfigurationNode> m_Root;
    bool m_IsBuilt;

    void AddPaths(std::vector<ParameterPair>& pairs, const std::set<std::string>& lockedParameters);
    void PrunePaths(std::vector<ParameterPair>& pairs);
    void ComputeConfigurationCounts();
    std::vector<uint64_t> PreprocessPairs(std::vector<ParameterPair>& pairs, const std::set<std::string>& lockedParameters,
        std::vector<uint64_t>& lockedLevels);
};

} // namespace ktt
