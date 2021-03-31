#include <TuningRunner/ConfigurationNode.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationNode::ConfigurationNode() :
    m_Parent(nullptr),
    m_Index(0),
    m_ConfigurationsCount(0)
{}

ConfigurationNode::ConfigurationNode(const ConfigurationNode& parent, const size_t index) :
    m_Parent(&parent),
    m_Index(index),
    m_ConfigurationsCount(0)
{}

void ConfigurationNode::AddPaths(const std::vector<size_t>& indices, const size_t indicesIndex,
    const std::vector<uint64_t>& levels, const size_t levelsIndex, const std::vector<uint64_t>& lockedLevels)
{
    if (HasBroadcastLevel(levels, levelsIndex))
    {
        for (auto& child : m_Children)
        {
            child->AddPaths(indices, indicesIndex, levels, levelsIndex, lockedLevels);
        }

        return;
    }

    const size_t index = indices[indicesIndex];

    if (!HasChildWithIndex(index))
    {
        const uint64_t level = GetLevel();

        if (ContainsElement(lockedLevels, level + 1))
        {
            return;
        }

        AddChild(index);
    }

    if (indices.size() > indicesIndex + 1)
    {
        auto& child = GetChildWithIndex(index);
        child.AddPaths(indices, indicesIndex + 1, levels, levelsIndex + 1, lockedLevels);
    }
}

void ConfigurationNode::PrunePaths(const std::vector<size_t>& indices, const size_t indicesIndex,
    const std::vector<uint64_t>& levels, const size_t levelsIndex)
{
    if (HasBroadcastLevel(levels, levelsIndex))
    {
        std::vector<ConfigurationNode*> toErase;

        for (auto& child : m_Children)
        {
            const size_t originalCount = child->GetChildrenCount();
            child->PrunePaths(indices, indicesIndex, levels, levelsIndex);

            if (originalCount > 0 && child->GetChildrenCount() == 0)
            {
                toErase.push_back(child.get());
            }
        }

        EraseIf(m_Children, [&toErase](const auto& currentChild)
        {
            return ContainsElement(toErase, currentChild.get());
        });

        return;
    }

    const size_t index = indices[indicesIndex];

    if (HasChildWithIndex(index))
    {
        auto& child = GetChildWithIndex(index);
        child.PrunePaths(indices, indicesIndex + 1, levels, levelsIndex + 1);

        if (child.GetChildrenCount() == 0)
        {
            EraseIf(m_Children, [&child](const auto& currentChild)
            {
                return &child == currentChild.get();
            });
        }
    }
}

void ConfigurationNode::ComputeConfigurationsCount()
{
    if (m_Children.empty())
    {
        m_ConfigurationsCount = 1;
        return;
    }

    uint64_t count = 0;

    for (const auto& child : m_Children)
    {
        child->ComputeConfigurationsCount();
        count += child->GetConfigurationsCount();
    }

    m_ConfigurationsCount = count;
}

void ConfigurationNode::GatherParameterIndices(const uint64_t index, std::vector<size_t>& indices) const
{
    if (m_Parent != nullptr)
    {
        indices.push_back(m_Index);
    }

    if (m_Children.empty())
    {
        KttAssert(index == 1, "Leaf nodes should only be referenced with index 1");
        return;
    }

    uint64_t cumulativeConfigurations = 0;

    for (const auto& child : m_Children)
    {
        const uint64_t skippedConfigurations = cumulativeConfigurations;
        cumulativeConfigurations += child->GetConfigurationsCount();

        if (index <= cumulativeConfigurations)
        {
            child->GatherParameterIndices(index - skippedConfigurations, indices);
            return;
        }
    }
}

std::vector<std::vector<size_t>> ConfigurationNode::GatherNeighbourIndices(const std::vector<size_t>& parameterIndices,
    const uint64_t maxDifferences, const size_t maxNeighbours, const std::set<std::vector<size_t>>& exploredIndices) const
{
    std::vector<std::vector<size_t>> result;
    std::vector<size_t> initialIndices;
    GatherNeighbours(result, initialIndices, parameterIndices, maxDifferences, maxNeighbours, exploredIndices);
    return result;
}

uint64_t ConfigurationNode::ComputeLocalIndex(const std::vector<size_t>& parameterIndices) const
{
    if (m_Children.empty())
    {
        return 1;
    }

    uint64_t result = 0;
    const size_t searchedIndex = parameterIndices[GetLevel()];

    for (const auto& child : m_Children)
    {
        if (child->GetIndex() == searchedIndex)
        {
            result += child->ComputeLocalIndex(parameterIndices);
            break;
        }
        
        result += child->GetConfigurationsCount();
    }

    return result;
}

const ConfigurationNode* ConfigurationNode::GetParent() const
{
    return m_Parent;
}

uint64_t ConfigurationNode::GetLevel() const
{
    uint64_t result = 0;
    const ConfigurationNode* parent = m_Parent;

    while (parent != nullptr)
    {
        parent = parent->GetParent();
        ++result;
    }

    return result;
}

size_t ConfigurationNode::GetIndex() const
{
    return m_Index;
}

size_t ConfigurationNode::GetChildrenCount() const
{
    return m_Children.size();
}

uint64_t ConfigurationNode::GetConfigurationsCount() const
{
    return m_ConfigurationsCount;
}

void ConfigurationNode::AddChild(const size_t index)
{
    m_Children.push_back(std::make_unique<ConfigurationNode>(*this, index));
}

bool ConfigurationNode::HasChildWithIndex(const size_t index) const
{
    return GetChildWithIndexPointer(index) != nullptr;
}

ConfigurationNode& ConfigurationNode::GetChildWithIndex(const size_t index) const
{
    auto* child = GetChildWithIndexPointer(index);
    KttAssert(child != nullptr, "No child with given index found");
    return *child;
}

ConfigurationNode* ConfigurationNode::GetChildWithIndexPointer(const size_t index) const
{
    for (auto& child : m_Children)
    {
        if (child->GetIndex() == index)
        {
            return child.get();
        }
    }

    return nullptr;
}

bool ConfigurationNode::HasBroadcastLevel(const std::vector<uint64_t>& levels, const size_t levelsIndex) const
{
    // Node has broadcast level if its level is not included in the levels vector.
    const uint64_t level = GetLevel();
    return levels.size() <= levelsIndex || levels[levelsIndex] - 1 != level;
}

void ConfigurationNode::GatherNeighbours(std::vector<std::vector<size_t>>& result, std::vector<size_t>& partialResult,
    const std::vector<size_t>& parameterIndices, const uint64_t maxDifferences, const size_t maxNeighbours,
    const std::set<std::vector<size_t>>& exploredIndices) const
{
    if (result.size() >= maxNeighbours)
    {
        return;
    }

    if (m_Children.empty())
    {
        if (!ContainsKey(exploredIndices, partialResult) && partialResult != parameterIndices)
        {
            result.push_back(partialResult);
        }

        return;
    }

    for (const auto& child : m_Children)
    {
        uint64_t newMaxDifferences = maxDifferences;

        if (child->GetIndex() != parameterIndices[GetLevel()])
        {
            if (maxDifferences == 0)
            {
                continue;
            }

            --newMaxDifferences;
        }

        std::vector<size_t> newPartialResult = partialResult;
        newPartialResult.push_back(child->GetIndex());
        child->GatherNeighbours(result, newPartialResult, parameterIndices, newMaxDifferences, maxNeighbours, exploredIndices);
    }
}

} // namespace ktt
