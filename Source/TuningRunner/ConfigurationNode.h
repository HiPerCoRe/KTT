#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Api/Configuration/ParameterPair.h>

namespace ktt
{

class ConfigurationNode
{
public:
    ConfigurationNode();
    ConfigurationNode(const ConfigurationNode& parent, const uint64_t value);

    void AddPaths(const std::vector<ParameterPair>& pairs, const size_t pairsIndex, const std::vector<uint64_t>& levels,
        const size_t levelsIndex, const std::vector<uint64_t>& lockedLevels);
    void PrunePaths(const std::vector<ParameterPair>& pairs, const size_t pairsIndex, const std::vector<uint64_t>& levels,
        const size_t levelsIndex);
    void ComputeConfigurationCounts();

    const ConfigurationNode* GetParent() const;
    uint64_t GetLevel() const;
    uint64_t GetValue() const;
    size_t GetChildrenCount() const;
    uint64_t GetConfigurationCount() const;

private:
    std::vector<std::unique_ptr<ConfigurationNode>> m_Children;
    const ConfigurationNode* m_Parent;
    uint64_t m_Value;
    uint64_t m_ConfigurationCount;

    void AddChild(const uint64_t value);
    bool HasChildWithValue(const uint64_t value);
    ConfigurationNode& GetChildWithValue(const uint64_t value);
};

} // namespace ktt
