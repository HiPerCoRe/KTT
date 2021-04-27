#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
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
    uint64_t GetLocalConfigurationIndex(const KernelConfiguration& configuration) const;
    bool IsConfigurationValid(const KernelConfiguration& configuration) const;

private:
    std::map<const KernelParameter*, uint64_t> m_ParameterToLevel;
    std::unique_ptr<ConfigurationNode> m_Root;
    bool m_IsBuilt;

    void InitializeParameterLevels(const std::vector<const KernelParameter*>& parameters);
    void AddPath(const std::vector<size_t>& indices);
    void RemovePath(const std::vector<size_t>& indices);
    KernelConfiguration GetConfigurationFromIndices(const std::vector<size_t>& indices) const;
    std::vector<size_t> GetIndicesFromConfiguration(const KernelConfiguration& configuration) const;
};

} // namespace ktt
