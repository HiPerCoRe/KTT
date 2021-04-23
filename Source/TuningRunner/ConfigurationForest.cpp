#include <Api/KttException.h>
#include <TuningRunner/ConfigurationForest.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

void ConfigurationForest::Build(const KernelParameterGroup& group, ctpl::thread_pool& pool)
{
    m_Subgroups = group.GenerateSubgroups();

    for (const auto& subgroup : m_Subgroups)
    {
        m_Trees.push_back(std::make_unique<ConfigurationTree>());
        auto& tree = *m_Trees.back();

        pool.push([&tree, &subgroup]()
        {
            tree.Build(subgroup);
        });
    }
}

void ConfigurationForest::Clear()
{
    m_Subgroups.clear();
    m_Trees.clear();
}

bool ConfigurationForest::IsBuilt() const
{
    bool result = true;

    for (const auto& tree : m_Trees)
    {
        result &= tree->IsBuilt();
    }
    
    return result;
}

bool ConfigurationForest::HasParameter(const std::string& name) const
{
    KttAssert(IsBuilt(), "The forest must be built before submitting queries");

    for (const auto& tree : m_Trees)
    {
        if (tree->HasParameter(name))
        {
            return true;
        }
    }

    return false;
}

uint64_t ConfigurationForest::GetConfigurationsCount() const
{
    KttAssert(IsBuilt(), "The forest must be built before submitting queries");
    uint64_t result = 1;

    for (const auto& tree : m_Trees)
    {
        result *= tree->GetConfigurationsCount();
    }

    return result;
}

KernelConfiguration ConfigurationForest::GetConfiguration(const uint64_t index) const
{
    KttAssert(IsBuilt(), "The forest must be built before submitting queries");

    if (index >= GetConfigurationsCount())
    {
        throw KttException("Invalid configuration index");
    }

    KernelConfiguration result;
    uint64_t curretIndex = index;

    for (const auto& tree : m_Trees)
    {
        const uint64_t localIndex = curretIndex % tree->GetConfigurationsCount();
        KernelConfiguration localConfiguration = tree->GetConfiguration(localIndex);
        result.Merge(localConfiguration);
        curretIndex /= tree->GetConfigurationsCount();
    }

    return result;
}

uint64_t ConfigurationForest::GetLocalConfigurationIndex(const KernelConfiguration& configuration) const
{
    KttAssert(IsBuilt(), "The forest must be built before submitting queries");
    uint64_t result = 0;
    uint64_t multiplier = 1;

    for (const auto& tree : m_Trees)
    {
        const uint64_t localIndex = tree->GetLocalConfigurationIndex(configuration);
        result += multiplier * localIndex;
        multiplier *= tree->GetConfigurationsCount();
    }
    
    return result;
}

bool ConfigurationForest::IsConfigurationValid(const KernelConfiguration& configuration) const
{
    KttAssert(IsBuilt(), "The forest must be built before submitting queries");
    bool result = true;

    for (const auto& tree : m_Trees)
    {
        result &= tree->IsConfigurationValid(configuration);
    }

    return result;
}

} // namespace ktt
