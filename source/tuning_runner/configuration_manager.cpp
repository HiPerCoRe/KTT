#include <limits>
#include <stdexcept>
#include "configuration_manager.h"
#include "searcher/annealing_searcher.h"
#include "searcher/full_searcher.h"
#include "searcher/random_searcher.h"
#include "searcher/mcmc_searcher.h"

namespace ktt
{

ConfigurationManager::ConfigurationManager() :
    searchMethod(SearchMethod::FullSearch)
{}

void ConfigurationManager::setKernelConfigurations(const KernelId id, const std::vector<KernelConfiguration>& configurations)
{
    clearKernelData(id, true, true);
    kernelConfigurations.insert(std::make_pair(id, configurations));
}

void ConfigurationManager::setKernelConfigurations(const KernelId id, const std::map<std::string, std::vector<KernelConfiguration>>& configurations)
{
    clearKernelData(id, true, true);
    packedKernelConfigurations.insert(std::make_pair(id, configurations));
}

void ConfigurationManager::setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
{
    if (method == SearchMethod::Annealing && arguments.size() < 1)
    {
        throw std::runtime_error(std::string("Insufficient number of arguments given for specified search method: ")
            + getSearchMethodName(method));
    }
    
    this->searchArguments = arguments;
    this->searchMethod = method;
}

bool ConfigurationManager::hasKernelConfigurations(const KernelId id) const
{
    return kernelConfigurations.find(id) != kernelConfigurations.end() || hasPackedConfigurations(id);
}

bool ConfigurationManager::hasPackedConfigurations(const KernelId id) const
{
    return packedKernelConfigurations.find(id) != packedKernelConfigurations.end();
}

void ConfigurationManager::clearKernelData(const KernelId id, const bool clearConfigurations, const bool clearBestConfiguration)
{
    if (searchers.find(id) != searchers.end())
    {
        searchers.erase(id);
    }

    if (clearConfigurations && kernelConfigurations.find(id) != kernelConfigurations.end())
    {
        kernelConfigurations.erase(id);
    }

    if (clearConfigurations && packedKernelConfigurations.find(id) != packedKernelConfigurations.end())
    {
        packedKernelConfigurations.erase(id);
        bestConfigurationsPerPack.erase(id);
    }

    if (clearBestConfiguration && bestConfigurations.find(id) != bestConfigurations.end())
    {
        bestConfigurations.erase(id);
    }
}

size_t ConfigurationManager::getConfigurationCount(const KernelId id)
{
    if (!hasPackedConfigurations(id))
    {
        auto configurations = kernelConfigurations.find(id);
        if (configurations != kernelConfigurations.end())
        {
            return configurations->second.size();
        }
    }
    else
    {
        auto packedConfigurations = packedKernelConfigurations.find(id);
        size_t totalCount = 0;
        for (const auto& pair : packedConfigurations->second)
        {
            totalCount += pair.second.size();
        }
        return totalCount;
    }

    return 0;
}

KernelConfiguration ConfigurationManager::getCurrentConfiguration(const KernelId id)
{
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        auto configurationPair = kernelConfigurations.find(id);
        if (configurationPair != kernelConfigurations.end())
        {
            initializeSearcher(id, searchMethod, searchArguments, configurationPair->second);
            searcherPair = searchers.find(id);
        }
        else
        {
            throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
        }
    }

    if (searcherPair->second->getUnexploredConfigurationCount() <= 0)
    {
        auto configurationPair = bestConfigurations.find(id);
        if (configurationPair == bestConfigurations.end())
        {
            throw std::runtime_error(std::string("No configurations left to explore and no best configuration recorded for kernel with id: ")
                + std::to_string(id));
        }
        return std::get<0>(configurationPair->second);
    }

    return searcherPair->second->getCurrentConfiguration();
}

KernelConfiguration ConfigurationManager::getBestConfiguration(const KernelId id)
{
    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        return getCurrentConfiguration(id);
    }

    return std::get<0>(configurationPair->second);
}

ComputationResult ConfigurationManager::getBestComputationResult(const KernelId id) const
{
    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        return ComputationResult("", std::vector<ParameterPair>{}, "Valid result does not exist");
    }

    return ComputationResult(std::get<1>(configurationPair->second), std::get<0>(configurationPair->second).getParameterPairs(),
        std::get<2>(configurationPair->second));
}

void ConfigurationManager::calculateNextConfiguration(const KernelId id, const std::string& kernelName, const KernelConfiguration& previous,
    const uint64_t previousDuration)
{
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        auto configurationPair = kernelConfigurations.find(id);
        if (configurationPair != kernelConfigurations.end())
        {
            initializeSearcher(id, searchMethod, searchArguments, configurationPair->second);
            searcherPair = searchers.find(id);
        }
        else
        {
            throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
        }
    }

    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, kernelName, previousDuration)));
    }
    else if (std::get<2>(configurationPair->second) > previousDuration)
    {
        bestConfigurations.erase(id);
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, kernelName, previousDuration)));
    }

    searcherPair->second->calculateNextConfiguration(static_cast<double>(previousDuration));
}

std::vector<KernelConfiguration> ConfigurationManager::fuseConfigurations(const KernelId id, const std::string& mainParameterPack)
{
    // todo
    return {};
}

void ConfigurationManager::initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
    const std::vector<KernelConfiguration>& configurations)
{
    switch (method)
    {
    case SearchMethod::FullSearch:
        searchers.insert(std::make_pair(id, std::make_unique<FullSearcher>(configurations)));
        break;
    case SearchMethod::RandomSearch:
        searchers.insert(std::make_pair(id, std::make_unique<RandomSearcher>(configurations)));
        break;
    case SearchMethod::Annealing:
        searchers.insert(std::make_pair(id, std::make_unique<AnnealingSearcher>(configurations, arguments.at(0))));
        break;
    case SearchMethod::MCMC:
        searchers.insert(std::make_pair(id, std::make_unique<MCMCSearcher>(configurations, arguments)));
        break;
    default:
        throw std::runtime_error("Specified searcher is not supported");
    }
}

std::string ConfigurationManager::getSearchMethodName(const SearchMethod method)
{
    switch (method)
    {
    case SearchMethod::FullSearch:
        return std::string("FullSearch");
    case SearchMethod::RandomSearch:
        return std::string("RandomSearch");
    case SearchMethod::Annealing:
        return std::string("Annealing");
    case SearchMethod::MCMC:
        return std::string("Markov chain Monte Carlo");
    default:
        return std::string("Unknown search method");
    }
}

} // namespace ktt
