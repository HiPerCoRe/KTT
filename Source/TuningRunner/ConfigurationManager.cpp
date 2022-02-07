#include <string>

#include <Api/Searcher/DeterministicSearcher.h>
#include <Api/KttException.h>
#include <TuningRunner/ConfigurationManager.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationManager::ConfigurationManager()
{
    Logger::LogDebug("Initializing configuration manager");
}

void ConfigurationManager::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    Logger::LogDebug("Adding new searcher for kernel with id " + std::to_string(id));
    ClearData(id);
    m_Searchers[id] = std::move(searcher);
}

void ConfigurationManager::InitializeData(const Kernel& kernel)
{
    const auto id = kernel.GetId();
    Logger::LogDebug("Initializing configuration data for kernel " + kernel.GetName());

    if (!ContainsKey(m_Searchers, id))
    {
        m_Searchers[id] = std::make_unique<DeterministicSearcher>();
    }

    m_ConfigurationData[id] = std::make_unique<ConfigurationData>(*m_Searchers[id], kernel);
}

void ConfigurationManager::ClearData(const KernelId id, const bool clearSearcher)
{
    Logger::LogDebug("Clearing configuration data for kernel with id " + std::to_string(id));
    m_ConfigurationData.erase(id);

    if (clearSearcher)
    {
        m_Searchers.erase(id);
    }
}

bool ConfigurationManager::CalculateNextConfiguration(const KernelId id, const KernelResult& previousResult)
{
    KttAssert(HasData(id), "Next configuration can only be calculated for kernels with initialized configuration data");
    return m_ConfigurationData[id]->CalculateNextConfiguration(previousResult);
}

void ConfigurationManager::ListConfigurations(const KernelId id) const
{
    KttAssert(HasData(id), "Configurations can only be listed for kernels with initialized configuration data");
    const auto& data = *m_ConfigurationData.find(id)->second;
    data.ListConfigurations();
}

bool ConfigurationManager::HasData(const KernelId id) const
{
    return ContainsKey(m_ConfigurationData, id) ;
}

bool ConfigurationManager::IsDataProcessed(const KernelId id) const
{
    if (!HasData(id))
    {
        return true;
    }

    return m_ConfigurationData.find(id)->second->IsProcessed();
}

uint64_t ConfigurationManager::GetTotalConfigurationsCount(const KernelId id) const
{
    if (!HasData(id))
    {
        return 0;
    }

    return m_ConfigurationData.find(id)->second->GetTotalConfigurationsCount();
}

uint64_t ConfigurationManager::GetExploredConfigurationsCount(const KernelId id) const
{
    if (!HasData(id))
    {
        return 0;
    }

    return m_ConfigurationData.find(id)->second->GetExploredConfigurationsCount();
}

KernelConfiguration ConfigurationManager::GetCurrentConfiguration(const KernelId id) const
{
    KttAssert(HasData(id), "Current configuration can only be retrieved for kernels with initialized configuration data");
    return m_ConfigurationData.find(id)->second->GetCurrentConfiguration();
}

KernelConfiguration ConfigurationManager::GetBestConfiguration(const KernelId id) const
{
    if (!HasData(id))
    {
        throw KttException("The best configuration can only be retrieved for kernels with initialized configuration data");
    }

    return m_ConfigurationData.find(id)->second->GetBestConfiguration();
}

} // namespace ktt
