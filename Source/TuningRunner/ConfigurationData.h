#pragma once

#include <map>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Info/DeviceInfo.h>
#include <Api/Searcher/Searcher.h>
#include <Kernel/Kernel.h>
#include <Kernel/KernelParameterGroup.h>
#include <KttTypes.h>

namespace ktt
{

class ConfigurationData
{
public:
    explicit ConfigurationData(const DeviceInfo& deviceInfo, Searcher& searcher, const Kernel& kernel);
    ~ConfigurationData();

    bool CalculateNextConfiguration(const KernelResult& previousResult);

    uint64_t GetConfigurationCount() const;
    bool IsProcessed() const;
    const KernelParameterGroup& GetCurrentGroup() const;
    const KernelConfiguration& GetCurrentConfiguration() const;
    const KernelConfiguration& GetBestConfiguration() const;

private:
    std::vector<KernelParameterGroup> m_Groups;
    std::vector<KernelConfiguration> m_Configurations;
    std::multimap<Nanoseconds, KernelConfiguration> m_ProcessedConfigurations;
    const DeviceInfo& m_DeviceInfo;
    Searcher& m_Searcher;
    const Kernel& m_Kernel;
    size_t m_CurrentGroup;

    bool InitializeNextGroup();
    void ComputeConfigurations(const KernelParameterGroup& group, const size_t currentIndex,
        std::vector<ParameterPair>& pairs, std::vector<KernelConfiguration>& finalResult) const;
    void AddExtraParameterPairs(std::vector<ParameterPair>& pairs) const;
    bool GetBestCompatibleConfiguration(const std::vector<ParameterPair>& pairs, KernelConfiguration& output) const;
    bool IsConfigurationCompatible(const std::vector<ParameterPair>& pairs, const KernelConfiguration& configuration) const;
    bool IsConfigurationValid(const std::vector<ParameterPair>& pairs) const;
    bool EvaluateConstraints(const std::vector<ParameterPair>& pairs) const;
};

} // namespace ktt
