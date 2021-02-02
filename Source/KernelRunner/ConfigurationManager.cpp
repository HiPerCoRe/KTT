#include <Api/Searcher/DeterministicSearcher.h>
#include <KernelRunner/ConfigurationManager.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ConfigurationManager::ConfigurationManager(const DeviceInfo& info) :
    m_DeviceInfo(info)
{}

void ConfigurationManager::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    ClearData(id);
    m_Searchers[id] = std::move(searcher);
}

void ConfigurationManager::InitializeData(const Kernel& kernel)
{
    const auto id = kernel.GetId();

    if (!ContainsKey(m_Searchers, id))
    {
        m_Searchers[id] = std::make_unique<DeterministicSearcher>();
    }

    m_ConfigurationData[id] = std::make_unique<ConfigurationData>(m_DeviceInfo, *m_Searchers[id], kernel);
}

void ConfigurationManager::ClearData(const KernelId id)
{
    m_ConfigurationData.erase(id);
}

void ConfigurationManager::CalculateNextConfiguration(const KernelId id, const KernelResult& previousResult)
{
    KttAssert(HasData(id), "Next configuration can only be calculated for kernels with initialized configuration data");
    m_ConfigurationData[id]->CalculateNextConfiguration(previousResult);
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

uint64_t ConfigurationManager::GetConfigurationCount(const KernelId id) const
{
    if (!HasData(id))
    {
        return 0;
    }

    return m_ConfigurationData.find(id)->second->GetConfigurationCount();
}

const KernelParameterGroup& ConfigurationManager::GetCurrentGroup(const KernelId id) const
{
    KttAssert(HasData(id), "Current group can only be retrieved for kernels with initialized configuration data");
    return m_ConfigurationData.find(id)->second->GetCurrentGroup();
}

const KernelConfiguration& ConfigurationManager::GetCurrentConfiguration(const KernelId id) const
{
    KttAssert(HasData(id), "Current configuration can only be retrieved for kernels with initialized configuration data");
    return m_ConfigurationData.find(id)->second->GetCurrentConfiguration();
}

const KernelConfiguration& ConfigurationManager::GetBestConfiguration(const KernelId id) const
{
    KttAssert(HasData(id), "Best configuration can only be retrieved for kernels with initialized configuration data");
    return m_ConfigurationData.find(id)->second->GetBestConfiguration();
}

} // namespace ktt
