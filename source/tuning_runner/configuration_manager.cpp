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

void ConfigurationManager::setKernelConfigurations(const KernelId id, const std::vector<KernelConfiguration>& configurations,
    const std::vector<KernelParameter>& parameters)
{
    clearData(id);
    initializeSearcher(id, searchMethod, searchArguments, configurations, parameters);
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
    return searchers.find(id) != searchers.end();
}

void ConfigurationManager::clearData(const KernelId id)
{
    clearSearcher(id);

    if (bestConfigurations.find(id) != bestConfigurations.end())
    {
        bestConfigurations.erase(id);
    }
}

void ConfigurationManager::clearSearcher(const KernelId id)
{
    if (searchers.find(id) != searchers.end())
    {
        searchers.erase(id);
    }
}

KernelConfiguration ConfigurationManager::getCurrentConfiguration(const KernelId id) const
{
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
    }

    if (searcherPair->second->getUnexploredConfigurationCount() <= 0)
    {
        auto configurationPair = bestConfigurations.find(id);
        if (configurationPair == bestConfigurations.end())
        {
            throw std::runtime_error(std::string("No configurations left to explore and no best configuration recorded for kernel with id: ")
                + std::to_string(id));
        }
        return configurationPair->second.first;
    }

    return searcherPair->second->getCurrentConfiguration();
}

KernelConfiguration ConfigurationManager::getBestConfiguration(const KernelId id) const
{
    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        return getCurrentConfiguration(id);
    }

    return configurationPair->second.first;
}

std::pair<std::vector<ParameterPair>, double> ConfigurationManager::getBestConfigurationPair(const KernelId id) const
{
    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        return std::make_pair(getCurrentConfiguration(id).getParameterPairs(), std::numeric_limits<double>::max());
    }

    return std::make_pair(configurationPair->second.first.getParameterPairs(), configurationPair->second.second);
}

void ConfigurationManager::calculateNextConfiguration(const KernelId id, const KernelConfiguration& previous, const double previousDuration)
{
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
    }

    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        bestConfigurations.insert(std::make_pair(id, std::make_pair(previous, previousDuration)));
    }
    else if (configurationPair->second.second > previousDuration)
    {
        bestConfigurations.erase(id);
        bestConfigurations.insert(std::make_pair(id, std::make_pair(previous, previousDuration)));
    }

    searcherPair->second->calculateNextConfiguration(previousDuration);
}

void ConfigurationManager::initializeSearcher(const KernelId id, const SearchMethod method, const std::vector<double>& arguments,
    const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters)
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
        searchers.insert(std::make_pair(id, std::make_unique<MCMCSearcher>(configurations, std::vector<double>(arguments.begin(),
            arguments.end()))));
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
