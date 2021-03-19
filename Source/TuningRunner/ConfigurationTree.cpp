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

        constraint.EnumerateParameterIndices([this, &processedParameters, &constraint, affectedCount](std::vector<size_t>& indices,
            const bool validIndices)
        {
            if (validIndices && affectedCount < constraint.GetParameterNames().size())
            {
                AddPaths(indices, constraint.GetParameters(), processedParameters);
            }
            else if (!validIndices && affectedCount > 0)
            {
                PrunePaths(indices, constraint.GetParameters());
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

        for (size_t index = 0; index < parameter->GetValuesCount(); ++index)
        {
            AddPaths({index}, {parameter}, processedParameters);
        }

        processedParameters.insert(parameter->GetName());
    }

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
    std::vector<ParameterPair> pairs;

    for (size_t i = 1; i < indices.size(); ++i)
    {
        for (const auto& pair : m_ParameterToLevel)
        {
            if (pair.second == i)
            {
                pairs.push_back(pair.first->GeneratePair(indices[i]));
                break;
            }
        }
    }

    return KernelConfiguration(pairs);
}

void ConfigurationTree::AddPaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters,
    std::set<std::string>& lockedParameters)
{
    std::vector<uint64_t> levels;
    const auto finalIndices = PreprocessIndices(indices, parameters, levels);
    const auto lockedLevels = GetLockedLevels(lockedParameters);
    m_Root->AddPaths(finalIndices, 0, levels, 0, lockedLevels);
}

void ConfigurationTree::PrunePaths(const std::vector<size_t>& indices, const std::vector<const KernelParameter*>& parameters)
{
    std::vector<uint64_t> levels;
    const auto finalIndices = PreprocessIndices(indices, parameters, levels);
    m_Root->PrunePaths(finalIndices, 0, levels, 0);
}

std::vector<size_t> ConfigurationTree::PreprocessIndices(const std::vector<size_t>& indices,
    const std::vector<const KernelParameter*>& parameters, std::vector<uint64_t>& levels)
{
    auto parameterIndices = MergeParametersWithIndices(parameters, indices);

    auto iterator = std::partition(parameterIndices.begin(), parameterIndices.end(), [this](const auto& pair)
    {
        return ContainsKey(m_ParameterToLevel, pair.first);
    });

    for (; iterator != parameterIndices.end(); ++iterator)
    {
        const uint64_t depth = GetDepth();
        m_ParameterToLevel[iterator->first] = depth + 1;
    }

    std::sort(parameterIndices.begin(), parameterIndices.end(), [this](const auto& leftPair, const auto& rightPair)
    {
        return m_ParameterToLevel[leftPair.first] < m_ParameterToLevel[rightPair.first];
    });

    std::vector<size_t> finalIndices;

    for (const auto& pair : parameterIndices)
    {
        finalIndices.push_back(pair.second);
        levels.push_back(m_ParameterToLevel[pair.first]);
    }

    return finalIndices;
}

std::vector<uint64_t> ConfigurationTree::GetLockedLevels(const std::set<std::string>& lockedParameters)
{
    std::vector<uint64_t> result;

    for (const auto& parameterName : lockedParameters)
    {
        for (const auto& pair : m_ParameterToLevel)
        {
            if (pair.first->GetName() == parameterName)
            {
                result.push_back(pair.second);
                break;
            }
        }
    }

    return result;
}

std::vector<std::pair<const KernelParameter*, size_t>> ConfigurationTree::MergeParametersWithIndices(
    const std::vector<const KernelParameter*>& parameters, const std::vector<size_t>& indices)
{
    KttAssert(parameters.size() == indices.size(), "Vector sizes must be equal");
    std::vector<std::pair<const KernelParameter*, size_t>> result;
    result.reserve(parameters.size());

    for (size_t i = 0; i < parameters.size(); ++i)
    {
        result.emplace_back(parameters[i], indices[i]);
    }

    return result;
}

} // namespace ktt
