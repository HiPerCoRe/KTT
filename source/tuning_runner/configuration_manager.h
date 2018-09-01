#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ktt_types.h"
#include "api/computation_result.h"
#include "enum/search_method.h"
#include "kernel/kernel_configuration.h"
#include "kernel/kernel_parameter.h"
#include "searcher/searcher.h"

namespace ktt
{

class ConfigurationManager
{
public:
    // Constructor
    ConfigurationManager();

    // Core methods
    void setKernelConfigurations(const KernelId id, const std::vector<KernelConfiguration>& configurations);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    bool hasKernelConfigurations(const KernelId id) const;
    void clearKernelData(const KernelId id, const bool clearConfigurations, const bool clearBestConfiguration);

    // Configuration search methods
    KernelConfiguration getCurrentConfiguration(const KernelId id);
    KernelConfiguration getBestConfiguration(const KernelId id);
    ComputationResult getBestComputationResult(const KernelId id) const;
    void calculateNextConfiguration(const KernelId id, const std::string& kernelName, const KernelConfiguration& previous,
        const uint64_t previousDuration);

private:
    // Attributes
    std::map<KernelId, std::vector<KernelConfiguration>> kernelConfigurations;
    std::map<KernelId, std::unique_ptr<Searcher>> searchers;
    std::map<KernelId, std::tuple<KernelConfiguration, std::string, uint64_t>> bestConfigurations;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    void initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
        const std::vector<KernelConfiguration>& configurations);
    static std::string getSearchMethodName(const SearchMethod method);
};

} // namespace ktt
