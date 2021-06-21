#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <vector>
#include <ctpl_stl.h>

#include <Api/Configuration/KernelConfiguration.h>
#include <Kernel/KernelParameterGroup.h>
#include <TuningRunner/ConfigurationTree.h>

namespace ktt
{

// Optimization wrapper over a configuration tree which is split into multiple trees by dividing its kernel parameter group into
// smaller groups based on constraint dependencies.
class ConfigurationForest
{
public:
    std::vector<std::future<void>> Build(const KernelParameterGroup& group, ctpl::thread_pool& pool);
    void Clear();

    bool IsBuilt() const;
    bool HasParameter(const std::string& name) const;
    uint64_t GetConfigurationsCount() const;
    KernelConfiguration GetConfiguration(const uint64_t index) const;
    uint64_t GetLocalConfigurationIndex(const KernelConfiguration& configuration) const;
    bool IsConfigurationValid(const KernelConfiguration& configuration) const;

private:
    std::vector<KernelParameterGroup> m_Subgroups;
    std::vector<std::unique_ptr<ConfigurationTree>> m_Trees;
};

} // namespace ktt
