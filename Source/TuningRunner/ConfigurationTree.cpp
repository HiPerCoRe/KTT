#include <Api/KttException.h>
#include <TuningRunner/ConfigurationTree.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

ConfigurationTree::ConfigurationTree() :
    m_Root(nullptr),
    m_IsBuilt(false)
{}

void ConfigurationTree::Build(const KernelParameterGroup& group)
{
    m_Root = std::make_unique<ConfigurationNode>();
    const auto orderedParameters = group.GetParametersInEnumerationOrder();
    InitializeParameterLevels(orderedParameters);

    group.EnumerateParameterIndices([this](const std::vector<size_t>& indices)
    {
        AddPath(indices);
    });

    m_Root->ComputeConfigurationsCount();
    m_IsBuilt = true;
}

void ConfigurationTree::Clear()
{
    m_ParameterToLevel.clear();
    m_Root.reset();
    m_IsBuilt = false;
}

bool ConfigurationTree::IsBuilt() const
{
    return m_IsBuilt;
}

bool ConfigurationTree::HasParameter(const std::string& name) const
{
    KttAssert(IsBuilt(), "The tree must be built before submitting queries");

    for (const auto& pair : m_ParameterToLevel)
    {
        if (pair.first->GetName() == name)
        {
            return true;
        }
    }

    return false;
}

uint64_t ConfigurationTree::GetDepth() const
{
    return m_ParameterToLevel.size();
}

uint64_t ConfigurationTree::GetConfigurationsCount() const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    return m_Root->GetConfigurationsCount();
}

KernelConfiguration ConfigurationTree::GetConfiguration(const uint64_t index) const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    
    if (index >= GetConfigurationsCount())
    {
        throw KttException("Invalid configuration index");
    }

    std::vector<size_t> indices;
    m_Root->GatherParameterIndices(index + 1, indices);
    return GetConfigurationFromIndices(indices);
}

uint64_t ConfigurationTree::GetLocalConfigurationIndex(const KernelConfiguration& configuration) const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    const std::vector<size_t> indices = GetIndicesFromConfiguration(configuration);
    uint64_t index = m_Root->ComputeLocalIndex(indices);
    --index;
    return index;
}

bool ConfigurationTree::IsConfigurationValid(const KernelConfiguration& configuration) const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    const std::vector<size_t> indices = GetIndicesFromConfiguration(configuration);
    return m_Root->IsPathValid(indices);
}

void ConfigurationTree::InitializeParameterLevels(const std::vector<const KernelParameter*>& parameters)
{
    for (const auto* parameter : parameters)
    {
        const uint64_t depth = GetDepth();
        m_ParameterToLevel[parameter] = depth + 1;
    }
}

void ConfigurationTree::AddPath(const std::vector<size_t>& indices)
{
    m_Root->AddPath(indices, 0);
}

void ConfigurationTree::RemovePath(const std::vector<size_t>& indices)
{
    m_Root->RemovePath(indices, 0);
}

KernelConfiguration ConfigurationTree::GetConfigurationFromIndices(const std::vector<size_t>& indices) const
{
    std::vector<ParameterPair> pairs;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        for (const auto& pair : m_ParameterToLevel)
        {
            if (pair.second == i + 1)
            {
                pairs.push_back(pair.first->GeneratePair(indices[i]));
                break;
            }
        }
    }

    return KernelConfiguration(pairs);
}

std::vector<size_t> ConfigurationTree::GetIndicesFromConfiguration(const KernelConfiguration& configuration) const
{
    std::vector<size_t> indices;

    for (uint64_t level = 1; level <= GetDepth(); ++level)
    {
        const KernelParameter* currentParameter = nullptr;

        for (const auto& pair : m_ParameterToLevel)
        {
            if (pair.second == level)
            {
                currentParameter = pair.first;
                break;
            }
        }

        for (const auto& pair : configuration.GetPairs())
        {
            if (pair.GetName() != currentParameter->GetName())
            {
                continue;
            }

            for (size_t index = 0; index < currentParameter->GetValuesCount(); ++index)
            {
                if (pair.HasSameValue(currentParameter->GeneratePair(index)))
                {
                    indices.push_back(index);
                    break;
                }
            }

            break;
        }
    }

    return indices;
}

} // namespace ktt
