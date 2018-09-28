#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ktt_types.h"
#include "api/computation_result.h"
#include "api/device_info.h"
#include "enum/search_method.h"
#include "kernel/kernel.h"
#include "kernel/kernel_composition.h"
#include "kernel/kernel_configuration.h"
#include "kernel/kernel_constraint.h"
#include "kernel/kernel_parameter.h"
#include "searcher/searcher.h"

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
    void calculateNextConfiguration(const Kernel& kernel, const KernelConfiguration& previous, const uint64_t previousDuration);
    void calculateNextConfiguration(const KernelComposition& composition, const KernelConfiguration& previous, const uint64_t previousDuration);

private:
    // Attributes
    std::map<KernelId, std::vector<KernelConfiguration>> kernelConfigurations;
    std::map<KernelId, std::pair<std::string, std::vector<KernelConfiguration>>> packKernelConfigurations;
    std::map<KernelId, std::vector<std::pair<size_t, std::string>>> orderedKernelPacks;
    mutable std::map<KernelId, size_t> currentPackIndices;
    std::map<KernelId, std::unique_ptr<Searcher>> searchers;
    std::map<KernelId, std::tuple<KernelConfiguration, std::string, uint64_t>> bestConfigurations;
    std::map<KernelId, std::map<std::string, std::tuple<KernelConfiguration, std::string, uint64_t>>> bestConfigurationsPerPack;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;
    DeviceInfo deviceInfo;
    static const std::string defaultParameterPackName;

    // Configuration generating methods
    std::vector<KernelConfiguration> getKernelConfigurations(const Kernel& kernel) const;
    std::vector<KernelConfiguration> getKernelCompositionConfigurations(const KernelComposition& composition) const;
    std::pair<std::string, std::vector<KernelConfiguration>> getNextPackKernelConfigurations(const Kernel& kernel) const;
    std::pair<std::string, std::vector<KernelConfiguration>> getNextPackKernelCompositionConfigurations(const KernelComposition& composition) const;

    // Helper methods
    void initializeOrderedKernelPacks(const Kernel& kernel);
    void initializeOrderedCompositionPacks(const KernelComposition& composition);
    void computeConfigurations(const Kernel& kernel, const std::vector<KernelParameter>& parameters, const std::vector<ParameterPair>& extraPairs,
        const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs, std::vector<KernelConfiguration>& finalResult) const;
    void computeCompositionConfigurations(const KernelComposition& composition, const std::vector<KernelParameter>& parameters,
        const std::vector<ParameterPair>& extraPairs, const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs,
        std::vector<KernelConfiguration>& finalResult) const;
    bool configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const;
    bool hasNextParameterPack(const KernelId id) const;
    std::string getNextParameterPack(const KernelId id) const;
    std::vector<ParameterPair> getExtraParameterPairs(const Kernel& kernel, const std::string& currentPack) const;
    std::vector<ParameterPair> getExtraParameterPairs(const KernelComposition& composition, const std::string& currentPack) const;
    void initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
        const std::vector<KernelConfiguration>& configurations);
    static size_t getConfigurationCountForParameters(const std::vector<KernelParameter>& parameters);
    static std::string getSearchMethodName(const SearchMethod method);
};

} // namespace ktt
