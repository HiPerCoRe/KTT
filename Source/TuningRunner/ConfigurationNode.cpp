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

void ConfigurationNode::AddPath(const std::vector<size_t>& indices, const size_t indicesIndex)
{
    const size_t index = indices[indicesIndex];

    if (!HasChildWithIndex(index))
    {
        AddChild(index);
    }

    if (indices.size() > indicesIndex + 1)
    {
        auto& child = GetChildWithIndex(index);
        child.AddPath(indices, indicesIndex + 1);
    }
}

void ConfigurationNode::RemovePath(const std::vector<size_t>& indices, const size_t indicesIndex)
{
    const size_t index = indices[indicesIndex];

    if (HasChildWithIndex(index))
    {
        auto& child = GetChildWithIndex(index);
        child.RemovePath(indices, indicesIndex + 1);

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

bool ConfigurationNode::IsPathValid(const std::vector<size_t>& parameterIndices) const
{
    if (m_Children.empty())
    {
        return parameterIndices.size() == GetLevel();
    }

    KttAssert(GetLevel() < parameterIndices.size(), "Invalid parameter indices input");
    const size_t searchedIndex = parameterIndices[GetLevel()];

    if (!HasChildWithIndex(searchedIndex))
    {
        return false;
    }

    const auto& child = GetChildWithIndex(searchedIndex);
    return child.IsPathValid(parameterIndices);
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

} // namespace ktt
