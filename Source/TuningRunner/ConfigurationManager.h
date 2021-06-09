#pragma once

#include <cstdint>
#include <map>
#include <memory>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <Api/Searcher/Searcher.h>
#include <Kernel/Kernel.h>
#include <TuningRunner/ConfigurationData.h>
#include <KttTypes.h>

namespace ktt
{

class ConfigurationManager
{
public:
    ConfigurationManager();

    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);
    void InitializeData(const Kernel& kernel);
    void ClearData(const KernelId id, const bool clearSearcher = false);
    bool CalculateNextConfiguration(const KernelId id, const KernelResult& previousResult);

    bool HasData(const KernelId id) const;
    bool IsDataProcessed(const KernelId id) const;
    uint64_t GetTotalConfigurationsCount(const KernelId id) const;
    uint64_t GetExploredConfigurationsCount(const KernelId id) const;
    KernelConfiguration GetCurrentConfiguration(const KernelId id) const;
    KernelConfiguration GetBestConfiguration(const KernelId id) const;

private:
    std::map<KernelId, std::unique_ptr<Searcher>> m_Searchers;
    std::map<KernelId, std::unique_ptr<ConfigurationData>> m_ConfigurationData;
};

} // namespace ktt
