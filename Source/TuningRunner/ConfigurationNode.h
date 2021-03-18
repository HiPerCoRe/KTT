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
    ConfigurationNode(const ConfigurationNode& parent, const size_t index);

    void AddPaths(const std::vector<size_t>& indices, const size_t indicesIndex, const std::vector<uint64_t>& levels,
        const size_t levelsIndex, const std::vector<uint64_t>& lockedLevels);
    void PrunePaths(const std::vector<size_t>& indices, const size_t indicesIndex, const std::vector<uint64_t>& levels,
        const size_t levelsIndex);
    void ComputeConfigurationCounts();

    const ConfigurationNode* GetParent() const;
    uint64_t GetLevel() const;
    size_t GetIndex() const;
    size_t GetChildrenCount() const;
    uint64_t GetConfigurationCount() const;
    void GatherParameterIndices(const uint64_t index, std::vector<size_t>& indices) const;

private:
    std::vector<std::unique_ptr<ConfigurationNode>> m_Children;
    const ConfigurationNode* m_Parent;
    size_t m_Index;
    uint64_t m_ConfigurationCount;

    void AddChild(const size_t index);
    bool HasChildWithIndex(const size_t index);
    ConfigurationNode& GetChildWithIndex(const size_t index);
};

} // namespace ktt
