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
    void setKernelConfigurations(const KernelId id, const std::vector<KernelConfiguration>& configurations,
        const std::vector<KernelParameter>& parameters);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    bool hasKernelConfigurations(const KernelId id) const;
    void clearData(const KernelId id);
    void clearSearcher(const KernelId id);

    // Configuration search methods
    KernelConfiguration getCurrentConfiguration(const KernelId id) const;
    KernelConfiguration getBestConfiguration(const KernelId id) const;
    ComputationResult getBestComputationResult(const KernelId id) const;
    void calculateNextConfiguration(const KernelId id, const std::string& kernelName, const KernelConfiguration& previous,
        const uint64_t previousDuration);

private:
    // Attributes
    std::map<KernelId, std::unique_ptr<Searcher>> searchers;
    std::map<KernelId, std::tuple<KernelConfiguration, std::string, uint64_t>> bestConfigurations;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    void initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters);
    static std::string getSearchMethodName(const SearchMethod method);
};

} // namespace ktt
