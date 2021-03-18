#include <algorithm>

#include <Api/KttException.h>
#include <TuningRunner/ConfigurationTree.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationTree::ConfigurationTree() :
    m_Root(nullptr),
    m_IsBuilt(false)
{}

void ConfigurationTree::Build(const KernelParameterGroup& group)
{
    m_Root = std::make_unique<ConfigurationNode>();

    std::set<const KernelConstraint*> processedConstraints;
    std::set<std::string> processedParameters;

    while (processedConstraints.size() != group.GetConstraints().size())
    {
        const KernelConstraint& constraint = group.GetNextConstraintToProcess(processedConstraints, processedParameters);
        const uint64_t affectedCount = constraint.GetAffectedParameterCount(processedParameters);

        constraint.EnumeratePairs([this, &processedParameters, &constraint, affectedCount](std::vector<ParameterPair>& pairs,
            const bool validPairs)
        {
            if (validPairs && affectedCount < constraint.GetParameterNames().size())
            {
                AddPaths(pairs, processedParameters);
            }
            else if (!validPairs && affectedCount > 0)
            {
                PrunePaths(pairs);
            }
        });

        processedConstraints.insert(&constraint);

        for (const auto& name : constraint.GetParameterNames())
        {
            processedParameters.insert(name);
        }
    }

    for (const auto* parameter : group.GetParameters())
    {
        if (ContainsKey(processedParameters, parameter->GetName()))
        {
            continue;
        }

        for (const auto& pair : parameter->GeneratePairs())
        {
            std::vector<ParameterPair> pairs{pair};
            AddPaths(pairs, processedParameters);
        }

        processedParameters.insert(parameter->GetName());
    }

    ComputeConfigurationCounts();
    m_IsBuilt = true;
}

void ConfigurationTree::Clear()
{
    m_ParameterToLevel.clear();
    m_Root.reset();
    m_IsBuilt = false;
}

uint64_t ConfigurationTree::GetDepth() const
{
    return m_ParameterToLevel.size();
}

uint64_t ConfigurationTree::GetConfigurationCount() const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    return m_Root->GetConfigurationCount();
}

KernelConfiguration ConfigurationTree::GetConfiguration(const uint64_t index) const
{
    KttAssert(m_IsBuilt, "The tree must be built before submitting queries");
    
    if (index >= GetConfigurationCount())
    {
        throw KttException("Invalid configuration index");
    }

    std::vector<uint64_t> values;
    m_Root->GatherValuesForIndex(index + 1, values);

    std::vector<ParameterPair> pairs;

    for (size_t i = 1; i < values.size(); ++i)
    {
        for (const auto& pair : m_ParameterToLevel)
        {
            if (pair.second == i)
            {
                pairs.emplace_back(pair.first, values[i]);
                break;
            }
        }
    }

    return KernelConfiguration(pairs);
}

void ConfigurationTree::AddPaths(std::vector<ParameterPair>& pairs, const std::set<std::string>& lockedParameters)
{
    std::vector<uint64_t> lockedLevels;
    const auto levels = PreprocessPairs(pairs, lockedParameters, lockedLevels);
    m_Root->AddPaths(pairs, 0, levels, 0, lockedLevels);
}

void ConfigurationTree::PrunePaths(std::vector<ParameterPair>& pairs)
{
    std::vector<uint64_t> lockedLevels;
    const auto levels = PreprocessPairs(pairs, {}, lockedLevels);
    m_Root->PrunePaths(pairs, 0, levels, 0);
}

void ConfigurationTree::ComputeConfigurationCounts()
{
    m_Root->ComputeConfigurationCounts();
}

std::vector<uint64_t> ConfigurationTree::PreprocessPairs(std::vector<ParameterPair>& pairs,
    const std::set<std::string>& lockedParameters, std::vector<uint64_t>& lockedLevels)
{
    auto iterator = std::partition(pairs.begin(), pairs.end(), [this](const auto& pair)
    {
        return ContainsKey(m_ParameterToLevel, pair.GetName());
    });

    for (; iterator != pairs.end(); ++iterator)
    {
        const uint64_t depth = GetDepth();
        m_ParameterToLevel[iterator->GetName()] = depth + 1;
    }

    std::sort(pairs.begin(), pairs.end(), [this](const auto& leftPair, const auto& rightPair)
    {
        return m_ParameterToLevel[leftPair.GetName()] < m_ParameterToLevel[rightPair.GetName()];
    });

    std::vector<uint64_t> parameterLevels;

    for (const auto& pair : pairs)
    {
        parameterLevels.push_back(m_ParameterToLevel[pair.GetName()]);
    }

    for (const auto& parameter : lockedParameters)
    {
        lockedLevels.push_back(m_ParameterToLevel[parameter]);
    }

    return parameterLevels;
}

} // namespace ktt
