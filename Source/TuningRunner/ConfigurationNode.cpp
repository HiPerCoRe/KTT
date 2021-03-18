#include <TuningRunner/ConfigurationNode.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationNode::ConfigurationNode() :
    m_Parent(nullptr),
    m_Value(0),
    m_ConfigurationCount(0)
{}

ConfigurationNode::ConfigurationNode(const ConfigurationNode& parent, const uint64_t value) :
    m_Parent(&parent),
    m_Value(value),
    m_ConfigurationCount(0)
{}

void ConfigurationNode::AddPaths(const std::vector<ParameterPair>& pairs, const size_t pairsIndex,
    const std::vector<uint64_t>& levels, const size_t levelsIndex, const std::vector<uint64_t>& lockedLevels)
{
    const uint64_t level = GetLevel();

    if (levels.size() <= levelsIndex || levels[levelsIndex] - 1 != level)
    {
        for (auto& child : m_Children)
        {
            child->AddPaths(pairs, pairsIndex, levels, levelsIndex, lockedLevels);
        }

        return;
    }

    const uint64_t value = pairs[pairsIndex].GetValue();

    if (!HasChildWithValue(value))
    {
        if (ContainsElement(lockedLevels, level + 1))
        {
            return;
        }

        AddChild(value);
    }

    if (pairs.size() > pairsIndex + 1)
    {
        auto& child = GetChildWithValue(value);
        child.AddPaths(pairs, pairsIndex + 1, levels, levelsIndex + 1, lockedLevels);
    }
}

void ConfigurationNode::PrunePaths(const std::vector<ParameterPair>& pairs, const size_t pairsIndex,
    const std::vector<uint64_t>& levels, const size_t levelsIndex)
{
    const uint64_t level = GetLevel();

    if (levels.size() <= levelsIndex || levels[levelsIndex] - 1 != level)
    {
        std::vector<ConfigurationNode*> toErase;

        for (auto& child : m_Children)
        {
            const size_t originalCount = child->GetChildrenCount();
            child->PrunePaths(pairs, pairsIndex, levels, levelsIndex);

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

    const uint64_t value = pairs[pairsIndex].GetValue();

    if (HasChildWithValue(value))
    {
        auto& child = GetChildWithValue(value);
        child.PrunePaths(pairs, pairsIndex + 1, levels, levelsIndex + 1);

        if (child.GetChildrenCount() == 0)
        {
            EraseIf(m_Children, [&child](const auto& currentChild)
            {
                return &child == currentChild.get();
            });
        }
    }
}

void ConfigurationNode::ComputeConfigurationCounts()
{
    if (m_Children.empty())
    {
        m_ConfigurationCount = 1;
        return;
    }

    uint64_t count = 0;

    for (const auto& child : m_Children)
    {
        child->ComputeConfigurationCounts();
        count += child->GetConfigurationCount();
    }

    m_ConfigurationCount = count;
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

uint64_t ConfigurationNode::GetValue() const
{
    return m_Value;
}

size_t ConfigurationNode::GetChildrenCount() const
{
    return m_Children.size();
}

uint64_t ConfigurationNode::GetConfigurationCount() const
{
    return m_ConfigurationCount;
}

void ConfigurationNode::GatherValuesForIndex(const uint64_t index, std::vector<uint64_t>& values) const
{
    values.push_back(m_Value);

    if (m_Children.empty())
    {
        KttAssert(index == 1, "Leaf nodes should only be referenced with index 1");
        return;
    }

    uint64_t cumulativeConfigurations = 0;
    uint64_t skippedConfigurations = 0;

    for (const auto& child : m_Children)
    {
        cumulativeConfigurations += child->GetConfigurationCount();

        if (index <= cumulativeConfigurations)
        {
            child->GatherValuesForIndex(index - skippedConfigurations, values);
            return;
        }

        skippedConfigurations = cumulativeConfigurations;
    }
}

void ConfigurationNode::AddChild(const uint64_t value)
{
    m_Children.push_back(std::make_unique<ConfigurationNode>(*this, value));
}

bool ConfigurationNode::HasChildWithValue(const uint64_t value)
{
    for (auto& child : m_Children)
    {
        if (child->GetValue() == value)
        {
            return true;
        }
    }

    return false;
}

ConfigurationNode& ConfigurationNode::GetChildWithValue(const uint64_t value)
{
    for (auto& child : m_Children)
    {
        if (child->GetValue() == value)
        {
            return *child;
        }
    }

    KttError("No child with given value found");
    return **m_Children.begin();
}

} // namespace ktt
