#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <api/computation_result.h>
#include <api/device_info.h>
#include <dto/kernel_result.h>
#include <enum/search_method.h>
#include <kernel/kernel.h>
#include <kernel/kernel_composition.h>
#include <kernel/kernel_configuration.h>
#include <kernel/kernel_constraint.h>
#include <kernel/kernel_parameter.h>
#include <tuning_runner/searcher/searcher.h>
#include <tuning_runner/configuration_storage.h>
#include <ktt_types.h>

namespace ktt
{

class ConfigurationManager
{
public:
    // Constructor
    ConfigurationManager(const DeviceInfo& info);

    // Core methods
    void initializeConfigurations(const Kernel& kernel);
    void initializeConfigurations(const KernelComposition& composition);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    bool hasKernelConfigurations(const KernelId id) const;
    bool hasPackConfigurations(const KernelId id) const;
    void clearKernelData(const KernelId id, const bool clearConfigurations, const bool clearBestConfiguration);

    // Configuration search methods
    size_t getConfigurationCount(const KernelId id);
    KernelConfiguration getCurrentConfiguration(const Kernel& kernel);
    KernelConfiguration getCurrentConfiguration(const KernelComposition& composition);
    KernelConfiguration getBestConfiguration(const Kernel& kernel);
    KernelConfiguration getBestConfiguration(const KernelComposition& composition);
    ComputationResult getBestComputationResult(const KernelId id) const;
    void calculateNextConfiguration(const Kernel& kernel, const KernelResult& previousResult);
    void calculateNextConfiguration(const KernelComposition& composition, const KernelResult& previousResult);

private:
    // Attributes
    std::map<KernelId, std::vector<KernelConfiguration>> kernelConfigurations;
    std::map<KernelId, std::pair<std::string, std::vector<KernelConfiguration>>> packKernelConfigurations;
    std::map<KernelId, std::vector<std::pair<size_t, std::string>>> orderedKernelPacks;
    mutable std::map<KernelId, size_t> currentPackIndices;
    std::map<KernelId, std::unique_ptr<Searcher>> searchers;
    std::map<KernelId, KernelResult> bestConfigurations;
    std::map<KernelId, ConfigurationStorage> configurationStorages;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;
    DeviceInfo deviceInfo;
    static const std::string defaultParameterPackName;

    // Helper methods
    void initializeOrderedKernelPacks(const Kernel& kernel);
    void initializeOrderedCompositionPacks(const KernelComposition& composition);
    void prepareNextPackKernelConfigurations(const Kernel& kernel);
    void prepareNextPackKernelCompositionConfigurations(const KernelComposition& composition);
    void computeConfigurations(const Kernel& kernel, const std::vector<KernelParameter>& parameters, const bool addExtraPairs,
        const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs, std::vector<KernelConfiguration>& finalResult) const;
    void computeCompositionConfigurations(const KernelComposition& composition, const std::vector<KernelParameter>& parameters,
        const bool addExtraPairs, const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs,
        std::vector<KernelConfiguration>& finalResult) const;
    bool configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const;
    bool hasNextParameterPack(const KernelId id) const;
    std::string getNextParameterPack(const KernelId id) const;
    std::vector<ParameterPair> getExtraParameterPairs(const Kernel& kernel, const KernelParameterPack& currentPack,
        const std::vector<ParameterPair>& generatedPairs) const;
    std::vector<ParameterPair> getExtraParameterPairs(const KernelComposition& composition, const KernelParameterPack& currentPack,
        const std::vector<ParameterPair>& generatedPairs) const;
    KernelParameterPack getCurrentParameterPack(const Kernel& kernel) const;
    KernelParameterPack getCurrentParameterPack(const KernelComposition& composition) const;
    void initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
        const std::vector<KernelConfiguration>& configurations);
    static bool checkParameterPairs(const std::vector<ParameterPair>& pairs, const std::vector<KernelConstraint>& constraints);
    static size_t getConfigurationCountForParameters(const std::vector<KernelParameter>& parameters);
    static std::string getSearchMethodName(const SearchMethod method);
};

} // namespace ktt
