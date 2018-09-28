#include <algorithm>
#include <limits>
#include <stdexcept>
#include "configuration_manager.h"
#include "searcher/annealing_searcher.h"
#include "searcher/full_searcher.h"
#include "searcher/random_searcher.h"
#include "searcher/mcmc_searcher.h"

namespace ktt
{

const std::string ConfigurationManager::defaultParameterPackName = "KTTStandaloneParameters";

ConfigurationManager::ConfigurationManager(const DeviceInfo& info) :
    searchMethod(SearchMethod::FullSearch),
    deviceInfo(info)
{}

void ConfigurationManager::initializeConfigurations(const Kernel& kernel)
{
    clearKernelData(kernel.getId(), true, true);

    if (kernel.getParameterPacks().empty())
    {
        std::vector<KernelConfiguration> configurations = getKernelConfigurations(kernel);
        kernelConfigurations.insert(std::make_pair(kernel.getId(), configurations));
    }
    else
    {
        initializeOrderedKernelPacks(kernel);
        std::pair<std::string, std::vector<KernelConfiguration>> packedConfigurations = getNextPackKernelConfigurations(kernel);
        packKernelConfigurations.insert(std::make_pair(kernel.getId(), packedConfigurations));
    }
}

void ConfigurationManager::initializeConfigurations(const KernelComposition& composition)
{
    clearKernelData(composition.getId(), true, true);

    if (composition.getParameterPacks().empty())
    {
        std::vector<KernelConfiguration> configurations = getKernelCompositionConfigurations(composition);
        kernelConfigurations.insert(std::make_pair(composition.getId(), configurations));
    }
    else
    {
        initializeOrderedCompositionPacks(composition);
        std::pair<std::string, std::vector<KernelConfiguration>> packConfigurations = getNextPackKernelCompositionConfigurations(composition);
        packKernelConfigurations.insert(std::make_pair(composition.getId(), packConfigurations));
    }
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
    return kernelConfigurations.find(id) != kernelConfigurations.end() || hasPackConfigurations(id);
}

bool ConfigurationManager::hasPackConfigurations(const KernelId id) const
{
    return orderedKernelPacks.find(id) != orderedKernelPacks.end();
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

    if (clearConfigurations && hasPackConfigurations(id))
    {
        packKernelConfigurations.erase(id);
        orderedKernelPacks.erase(id);
        currentPackIndices.erase(id);
        bestConfigurationsPerPack.erase(id);
    }

    if (clearBestConfiguration && bestConfigurations.find(id) != bestConfigurations.end())
    {
        bestConfigurations.erase(id);
    }
}

size_t ConfigurationManager::getConfigurationCount(const KernelId id)
{
    if (!hasPackConfigurations(id))
    {
        auto configurations = kernelConfigurations.find(id);
        if (configurations != kernelConfigurations.end())
        {
            return configurations->second.size();
        }
    }
    else
    {
        auto orderedPacks = orderedKernelPacks.find(id);
        if (orderedPacks != orderedKernelPacks.end())
        {
            size_t totalCount = 0;
            for (const auto& pack : orderedPacks->second)
            {
                totalCount += pack.first;
            }
            return totalCount;
        }
    }

    return 0;
}

KernelConfiguration ConfigurationManager::getCurrentConfiguration(const Kernel& kernel)
{
    const size_t id = kernel.getId();
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        if (!hasPackConfigurations(id))
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
        else
        {
            auto configurationPair = packKernelConfigurations.find(id);
            if (configurationPair != packKernelConfigurations.end())
            {
                initializeSearcher(id, searchMethod, searchArguments, configurationPair->second.second);
                searcherPair = searchers.find(id);
            }
            else
            {
                throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
            }
        }
    }

    if (searcherPair->second->getUnexploredConfigurationCount() <= 0)
    {
        if (!hasPackConfigurations(id) || !hasNextParameterPack(id))
        {
            auto configurationPair = bestConfigurations.find(id);
            if (configurationPair == bestConfigurations.end())
            {
                throw std::runtime_error(std::string("No configurations left to explore and no best configuration recorded for kernel with id: ")
                    + std::to_string(id));
            }
            return std::get<0>(configurationPair->second);
        }
        else
        {
            searchers.erase(id);
            packKernelConfigurations.erase(id);
            std::pair<std::string, std::vector<KernelConfiguration>> packedConfigurations = getNextPackKernelConfigurations(kernel);
            packKernelConfigurations.insert(std::make_pair(id, packedConfigurations));
            initializeSearcher(id, searchMethod, searchArguments, packKernelConfigurations.find(id)->second.second);
            searcherPair = searchers.find(id);
        }
    }

    return searcherPair->second->getCurrentConfiguration();
}

KernelConfiguration ConfigurationManager::getCurrentConfiguration(const KernelComposition& composition)
{
    const size_t id = composition.getId();
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        if (!hasPackConfigurations(id))
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
        else
        {
            auto configurationPair = packKernelConfigurations.find(id);
            if (configurationPair != packKernelConfigurations.end())
            {
                initializeSearcher(id, searchMethod, searchArguments, configurationPair->second.second);
                searcherPair = searchers.find(id);
            }
            else
            {
                throw std::runtime_error(std::string("Configuration for kernel with following id is not present: ") + std::to_string(id));
            }
        }
    }

    if (searcherPair->second->getUnexploredConfigurationCount() <= 0)
    {
        if (!hasPackConfigurations(id) || !hasNextParameterPack(id))
        {
            auto configurationPair = bestConfigurations.find(id);
            if (configurationPair == bestConfigurations.end())
            {
                throw std::runtime_error(std::string("No configurations left to explore and no best configuration recorded for kernel with id: ")
                    + std::to_string(id));
            }
            return std::get<0>(configurationPair->second);
        }
        else
        {
            searchers.erase(id);
            packKernelConfigurations.erase(id);
            std::pair<std::string, std::vector<KernelConfiguration>> packedConfigurations = getNextPackKernelCompositionConfigurations(composition);
            packKernelConfigurations.insert(std::make_pair(id, packedConfigurations));
            initializeSearcher(id, searchMethod, searchArguments, packKernelConfigurations.find(id)->second.second);
            searcherPair = searchers.find(id);
        }
    }

    return searcherPair->second->getCurrentConfiguration();
}

KernelConfiguration ConfigurationManager::getBestConfiguration(const Kernel& kernel)
{
    auto configurationPair = bestConfigurations.find(kernel.getId());
    if (configurationPair == bestConfigurations.end())
    {
        return getCurrentConfiguration(kernel);
    }

    return std::get<0>(configurationPair->second);
}

KernelConfiguration ConfigurationManager::getBestConfiguration(const KernelComposition& composition)
{
    auto configurationPair = bestConfigurations.find(composition.getId());
    if (configurationPair == bestConfigurations.end())
    {
        return getCurrentConfiguration(composition);
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

void ConfigurationManager::calculateNextConfiguration(const Kernel& kernel, const KernelConfiguration& previous, const uint64_t previousDuration)
{
    const size_t id = kernel.getId();
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        throw std::runtime_error(std::string("Configurations for the following kernel were not initialized yet: ") + kernel.getName());
    }

    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, kernel.getName(), previousDuration)));
    }
    else if (std::get<2>(configurationPair->second) > previousDuration)
    {
        bestConfigurations.erase(id);
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, kernel.getName(), previousDuration)));
    }

    if (hasPackConfigurations(id))
    {
        size_t targetPackIndex = currentPackIndices.find(id)->second - 1;
        const std::string targetPack = orderedKernelPacks.find(id)->second.at(targetPackIndex).second;

        auto configurationPerPackPair = bestConfigurationsPerPack.find(id);
        if (configurationPerPackPair == bestConfigurationsPerPack.end())
        {
            bestConfigurationsPerPack.insert(std::make_pair(id, std::map<std::string, std::tuple<KernelConfiguration, std::string, uint64_t>>{}));
        }
        configurationPerPackPair = bestConfigurationsPerPack.find(id);

        auto tagetPackPair = configurationPerPackPair->second.find(targetPack);
        if (tagetPackPair == configurationPerPackPair->second.end())
        {
            configurationPerPackPair->second.insert(std::make_pair(targetPack, std::make_tuple(previous, kernel.getName(), previousDuration)));
        }
        else if (std::get<2>(tagetPackPair->second) > previousDuration)
        {
            configurationPerPackPair->second.erase(targetPack);
            configurationPerPackPair->second.insert(std::make_pair(targetPack, std::make_tuple(previous, kernel.getName(), previousDuration)));
        }
    }

    searcherPair->second->calculateNextConfiguration(static_cast<double>(previousDuration));
}

void ConfigurationManager::calculateNextConfiguration(const KernelComposition& composition, const KernelConfiguration& previous,
    const uint64_t previousDuration)
{
    const size_t id = composition.getId();
    auto searcherPair = searchers.find(id);
    if (searcherPair == searchers.end())
    {
        throw std::runtime_error(std::string("Configurations for the following kernel composition were not initialized yet: ")
            + composition.getName());
    }

    auto configurationPair = bestConfigurations.find(id);
    if (configurationPair == bestConfigurations.end())
    {
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, composition.getName(), previousDuration)));
    }
    else if (std::get<2>(configurationPair->second) > previousDuration)
    {
        bestConfigurations.erase(id);
        bestConfigurations.insert(std::make_pair(id, std::make_tuple(previous, composition.getName(), previousDuration)));
    }

    if (hasPackConfigurations(id))
    {
        size_t targetPackIndex = currentPackIndices.find(id)->second - 1;
        const std::string targetPack = orderedKernelPacks.find(id)->second.at(targetPackIndex).second;

        auto configurationPerPackPair = bestConfigurationsPerPack.find(id);
        if (configurationPerPackPair == bestConfigurationsPerPack.end())
        {
            bestConfigurationsPerPack.insert(std::make_pair(id, std::map<std::string, std::tuple<KernelConfiguration, std::string, uint64_t>>{}));
        }
        configurationPerPackPair = bestConfigurationsPerPack.find(id);

        auto tagetPackPair = configurationPerPackPair->second.find(targetPack);
        if (tagetPackPair == configurationPerPackPair->second.end())
        {
            configurationPerPackPair->second.insert(std::make_pair(targetPack, std::make_tuple(previous, composition.getName(), previousDuration)));
        }
        else if (std::get<2>(tagetPackPair->second) > previousDuration)
        {
            configurationPerPackPair->second.erase(targetPack);
            configurationPerPackPair->second.insert(std::make_pair(targetPack, std::make_tuple(previous, composition.getName(), previousDuration)));
        }
    }

    searcherPair->second->calculateNextConfiguration(static_cast<double>(previousDuration));
}

std::vector<KernelConfiguration> ConfigurationManager::getKernelConfigurations(const Kernel& kernel) const
{
    std::vector<KernelConfiguration> configurations;
    computeConfigurations(kernel, kernel.getParameters(), std::vector<ParameterPair>{}, 0, std::vector<ParameterPair>{}, configurations);
    return configurations;
}

std::vector<KernelConfiguration> ConfigurationManager::getKernelCompositionConfigurations(const KernelComposition& composition) const
{
    std::vector<KernelConfiguration> kernelConfigurations;
    computeCompositionConfigurations(composition, composition.getParameters(), std::vector<ParameterPair>{}, 0, std::vector<ParameterPair>{},
        kernelConfigurations);
    return kernelConfigurations;
}

std::pair<std::string, std::vector<KernelConfiguration>> ConfigurationManager::getNextPackKernelConfigurations(const Kernel& kernel) const
{
    std::string nextPack = getNextParameterPack(kernel.getId());

    std::vector<KernelParameter> packParameters;
    if (nextPack == defaultParameterPackName)
    {
        packParameters = kernel.getParametersOutsidePacks();
    }
    else
    {
        packParameters = kernel.getParametersForPack(nextPack);
    }
    std::vector<ParameterPair> extraPairs = getExtraParameterPairs(kernel, nextPack);
    std::vector<KernelConfiguration> configurations;
    computeConfigurations(kernel, packParameters, extraPairs, 0, std::vector<ParameterPair>{}, configurations);

    return std::make_pair(nextPack, configurations);
}

std::pair<std::string, std::vector<KernelConfiguration>> ConfigurationManager::getNextPackKernelCompositionConfigurations(
    const KernelComposition& composition) const
{
    std::string nextPack = getNextParameterPack(composition.getId());

    std::vector<KernelParameter> packParameters;
    if (nextPack == defaultParameterPackName)
    {
        packParameters = composition.getParametersOutsidePacks();
    }
    else
    {
        packParameters = composition.getParametersForPack(nextPack);
    }
    std::vector<ParameterPair> extraPairs = getExtraParameterPairs(composition, nextPack);
    std::vector<KernelConfiguration> configurations;
    computeCompositionConfigurations(composition, packParameters, extraPairs, 0, std::vector<ParameterPair>{}, configurations);

    return std::make_pair(nextPack, configurations);
}

void ConfigurationManager::initializeOrderedKernelPacks(const Kernel& kernel)
{
    std::vector<std::pair<size_t, std::string>> orderedPacks;
    std::vector<KernelParameterPack> kernelPacks = kernel.getParameterPacks();
    
    std::vector<KernelParameter> defaultParameters = kernel.getParametersOutsidePacks();
    const size_t defaultParametersConfigurationCount = getConfigurationCountForParameters(defaultParameters);
    bool defaultPackProcessed = false;
    size_t orderedPacksCount = kernelPacks.size() + 1;

    if (defaultParameters.empty())
    {
        defaultPackProcessed = true;
        --orderedPacksCount;
    }

    for (size_t i = 0; i < orderedPacksCount; ++i)
    {
        std::string bestPack = defaultParameterPackName;
        size_t bestConfigurationCount = defaultParametersConfigurationCount;

        for (const auto& pack : kernelPacks)
        {
            bool packProcessed = false;
            for (const auto& orderedPack : orderedPacks)
            {
                if (orderedPack.second == pack.getName())
                {
                    packProcessed = true;
                    break;
                }
            }

            if (packProcessed)
            {
                continue;
            }

            std::vector<KernelParameter> currentParameters = kernel.getParametersForPack(pack);
            const size_t currentConfigurationCount = getConfigurationCountForParameters(currentParameters);

            if (bestConfigurationCount > currentConfigurationCount || bestPack == defaultParameterPackName && defaultPackProcessed)
            {
                bestConfigurationCount = currentConfigurationCount;
                bestPack = pack.getName();
            }
        }

        orderedPacks.push_back(std::make_pair(bestConfigurationCount, bestPack));
        if (bestPack == defaultParameterPackName)
        {
            defaultPackProcessed = true;
        }
    }

    currentPackIndices.insert(std::make_pair(kernel.getId(), 0));
    orderedKernelPacks.insert(std::make_pair(kernel.getId(), orderedPacks));
}

void ConfigurationManager::initializeOrderedCompositionPacks(const KernelComposition& composition)
{
    std::vector<std::pair<size_t, std::string>> orderedPacks;
    std::vector<KernelParameterPack> compositionPacks = composition.getParameterPacks();

    std::vector<KernelParameter> defaultParameters = composition.getParametersOutsidePacks();
    const size_t defaultParametersConfigurationCount = getConfigurationCountForParameters(defaultParameters);
    bool defaultPackProcessed = false;
    size_t orderedPacksCount = compositionPacks.size() + 1;

    if (defaultParameters.empty())
    {
        defaultPackProcessed = true;
        --orderedPacksCount;
    }

    for (size_t i = 0; i < orderedPacksCount; ++i)
    {
        std::string bestPack = defaultParameterPackName;
        size_t bestConfigurationCount = defaultParametersConfigurationCount;

        for (const auto& pack : compositionPacks)
        {
            bool packProcessed = false;
            for (const auto& orderedPack : orderedPacks)
            {
                if (orderedPack.second == pack.getName())
                {
                    packProcessed = true;
                    break;
                }
            }

            if (packProcessed)
            {
                continue;
            }

            std::vector<KernelParameter> currentParameters = composition.getParametersForPack(pack);
            const size_t currentConfigurationCount = getConfigurationCountForParameters(currentParameters);

            if (bestConfigurationCount > currentConfigurationCount || bestPack == defaultParameterPackName && defaultPackProcessed)
            {
                bestConfigurationCount = currentConfigurationCount;
                bestPack = pack.getName();
            }
        }

        orderedPacks.push_back(std::make_pair(bestConfigurationCount, bestPack));
        if (bestPack == defaultParameterPackName)
        {
            defaultPackProcessed = true;
        }
    }

    currentPackIndices.insert(std::make_pair(composition.getId(), 0));
    orderedKernelPacks.insert(std::make_pair(composition.getId(), orderedPacks));
}

void ConfigurationManager::computeConfigurations(const Kernel& kernel, const std::vector<KernelParameter>& parameters,
    const std::vector<ParameterPair>& extraPairs, const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        std::vector<ParameterPair> allPairs;
        allPairs.reserve(parameterPairs.size() + extraPairs.size());
        allPairs.insert(allPairs.end(), parameterPairs.begin(), parameterPairs.end());
        allPairs.insert(allPairs.end(), extraPairs.begin(), extraPairs.end());

        DimensionVector finalGlobalSize = kernel.getModifiedGlobalSize(allPairs);
        DimensionVector finalLocalSize = kernel.getModifiedLocalSize(allPairs);
        std::vector<LocalMemoryModifier> memoryModifiers = kernel.getLocalMemoryModifiers(allPairs);

        KernelConfiguration configuration(finalGlobalSize, finalLocalSize, allPairs, memoryModifiers);
        if (configurationIsValid(configuration, kernel.getConstraints()))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter

    if (!parameter.hasValuesDouble())
    {
        for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
        {
            std::vector<ParameterPair> newParameterPairs = parameterPairs;
            newParameterPairs.emplace_back(parameter.getName(), value);
            computeConfigurations(kernel, parameters, extraPairs, currentParameterIndex + 1, newParameterPairs, finalResult);
        }
    }
    else
    {
        for (const auto& value : parameter.getValuesDouble()) // recursively build tree of configurations for each parameter value
        {
            std::vector<ParameterPair> newParameterPairs = parameterPairs;
            newParameterPairs.emplace_back(parameter.getName(), value);
            computeConfigurations(kernel, parameters, extraPairs, currentParameterIndex + 1, newParameterPairs, finalResult);
        }
    }
}

void ConfigurationManager::computeCompositionConfigurations(const KernelComposition& composition, const std::vector<KernelParameter>& parameters,
    const std::vector<ParameterPair>& extraPairs, const size_t currentParameterIndex, const std::vector<ParameterPair>& parameterPairs,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        std::vector<ParameterPair> allPairs;
        allPairs.reserve(parameterPairs.size() + extraPairs.size());
        allPairs.insert(allPairs.end(), parameterPairs.begin(), parameterPairs.end());
        allPairs.insert(allPairs.end(), extraPairs.begin(), extraPairs.end());

        std::map<KernelId, DimensionVector> globalSizes = composition.getModifiedGlobalSizes(allPairs);
        std::map<KernelId, DimensionVector> localSizes = composition.getModifiedLocalSizes(allPairs);
        std::map<KernelId, std::vector<LocalMemoryModifier>> modifiers = composition.getLocalMemoryModifiers(allPairs);

        KernelConfiguration configuration(globalSizes, localSizes, allPairs, modifiers);
        if (configurationIsValid(configuration, composition.getConstraints()))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter

    if (!parameter.hasValuesDouble())
    {
        for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
        {
            std::vector<ParameterPair> newParameterPairs = parameterPairs;
            newParameterPairs.emplace_back(parameter.getName(), value);
            computeCompositionConfigurations(composition, parameters, extraPairs, currentParameterIndex + 1, newParameterPairs, finalResult);
        }
    }
    else
    {
        for (const auto& value : parameter.getValuesDouble()) // recursively build tree of configurations for each parameter value
        {
            std::vector<ParameterPair> newParameterPairs = parameterPairs;
            newParameterPairs.emplace_back(parameter.getName(), value);
            computeCompositionConfigurations(composition, parameters, extraPairs, currentParameterIndex + 1, newParameterPairs, finalResult);
        }
    }
}

bool ConfigurationManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const
{
    for (const auto& constraint : constraints)
    {
        std::vector<std::string> constraintNames = constraint.getParameterNames();
        std::vector<size_t> constraintValues(constraintNames.size());

        for (size_t i = 0; i < constraintNames.size(); i++)
        {
            for (const auto& parameterPair : configuration.getParameterPairs())
            {
                if (parameterPair.getName() == constraintNames.at(i))
                {
                    constraintValues.at(i) = parameterPair.getValue();
                    break;
                }
            }
        }

        auto constraintFunction = constraint.getConstraintFunction();
        if (!constraintFunction(constraintValues))
        {
            return false;
        }
    }

    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();
    for (const auto& localSize : localSizes)
    {
        if (localSize.getTotalSize() > deviceInfo.getMaxWorkGroupSize())
        {
            return false;
        }
    }

    return true;
}

bool ConfigurationManager::hasNextParameterPack(const KernelId id) const
{
    auto index = currentPackIndices.find(id);
    auto orderedPacks = orderedKernelPacks.find(id);

    if (index == currentPackIndices.end() || orderedPacks == orderedKernelPacks.end())
    {
        return false;
    }

    return index->second < orderedPacks->second.size();
}

std::string ConfigurationManager::getNextParameterPack(const KernelId id) const
{
    auto index = currentPackIndices.find(id);
    auto orderedPacks = orderedKernelPacks.find(id);

    if (index == currentPackIndices.end() || orderedPacks == orderedKernelPacks.end())
    {
        throw std::runtime_error(std::string("Parameter pack information not present for kernel with id: ") + std::to_string(id));
    }

    if (index->second >= orderedPacks->second.size())
    {
        throw std::runtime_error("All parameter packs were already processed");
    }

    std::string result = orderedPacks->second.at(index->second).second;
    ++index->second;

    return result;
}

std::vector<ParameterPair> ConfigurationManager::getExtraParameterPairs(const Kernel& kernel, const std::string& currentPack) const
{
    std::vector<ParameterPair> result;
    std::vector<KernelParameter> addedParameters;
    if (currentPack == defaultParameterPackName)
    {
        addedParameters = kernel.getParametersOutsidePacks();
    }
    else
    {
        addedParameters = kernel.getParametersForPack(currentPack);
    }
    std::vector<KernelParameter> allParameters = kernel.getParameters();

    auto configurationsPerPack = bestConfigurationsPerPack.find(kernel.getId());
    std::vector<KernelParameterPack> packs = kernel.getParameterPacks();
    if (currentPack != defaultParameterPackName)
    {
        std::vector<std::string> defaultParameterNames;
        std::vector<KernelParameter> defaultParameters = kernel.getParametersOutsidePacks();
        for (const auto& defaultParameter : defaultParameters)
        {
            defaultParameterNames.push_back(defaultParameter.getName());
        }

        packs.push_back(KernelParameterPack(defaultParameterPackName, defaultParameterNames));
    }

    if (configurationsPerPack != bestConfigurationsPerPack.end())
    {
        for (const auto& pack : packs)
        {
            if (pack.getName() == currentPack)
            {
                continue;
            }

            auto bestPackConfiguration = configurationsPerPack->second.find(pack.getName());

            if (bestPackConfiguration != configurationsPerPack->second.end())
            {
                std::vector<ParameterPair> bestParameterPairs = std::get<0>(bestPackConfiguration->second).getParameterPairs();
                for (const auto& bestPair : bestParameterPairs)
                {
                    bool alreadyAdded = false;
                    for (const auto& addedParameter : addedParameters)
                    {
                        if (addedParameter.getName() == bestPair.getName())
                        {
                            alreadyAdded = true;
                            break;
                        }
                    }

                    if (!alreadyAdded)
                    {
                        result.push_back(bestPair);
                        addedParameters.push_back(KernelParameter(bestPair.getName(), std::vector<size_t>{bestPair.getValue()}));
                    }
                }
            }
        }
    }

    for (const auto& parameter : allParameters)
    {
        bool alreadyAdded = false;
        for (const auto& addedParameter : addedParameters)
        {
            if (addedParameter == parameter)
            {
                alreadyAdded = true;
                break;
            }
        }

        if (!alreadyAdded)
        {
            result.push_back(ParameterPair(parameter.getName(), parameter.getValues().at(0)));
            addedParameters.push_back(parameter);
        }
    }

    return result;
}

std::vector<ParameterPair> ConfigurationManager::getExtraParameterPairs(const KernelComposition& composition, const std::string& currentPack) const
{
    std::vector<ParameterPair> result;
    std::vector<KernelParameter> addedParameters;
    if (currentPack == defaultParameterPackName)
    {
        addedParameters = composition.getParametersOutsidePacks();
    }
    else
    {
        addedParameters = composition.getParametersForPack(currentPack);
    }
    std::vector<KernelParameter> allParameters = composition.getParameters();

    auto configurationsPerPack = bestConfigurationsPerPack.find(composition.getId());
    std::vector<KernelParameterPack> packs = composition.getParameterPacks();
    if (currentPack != defaultParameterPackName)
    {
        std::vector<std::string> defaultParameterNames;
        std::vector<KernelParameter> defaultParameters = composition.getParametersOutsidePacks();
        for (const auto& defaultParameter : defaultParameters)
        {
            defaultParameterNames.push_back(defaultParameter.getName());
        }

        packs.push_back(KernelParameterPack(defaultParameterPackName, defaultParameterNames));
    }

    if (configurationsPerPack != bestConfigurationsPerPack.end())
    {
        for (const auto& pack : packs)
        {
            if (pack.getName() == currentPack)
            {
                continue;
            }

            auto bestPackConfiguration = configurationsPerPack->second.find(pack.getName());

            if (bestPackConfiguration != configurationsPerPack->second.end())
            {
                std::vector<ParameterPair> bestParameterPairs = std::get<0>(bestPackConfiguration->second).getParameterPairs();
                for (const auto& bestPair : bestParameterPairs)
                {
                    bool alreadyAdded = false;
                    for (const auto& addedParameter : addedParameters)
                    {
                        if (addedParameter.getName() == bestPair.getName())
                        {
                            alreadyAdded = true;
                            break;
                        }
                    }

                    if (!alreadyAdded)
                    {
                        result.push_back(bestPair);
                        addedParameters.push_back(KernelParameter(bestPair.getName(), std::vector<size_t>{bestPair.getValue()}));
                    }
                }
            }
        }
    }

    for (const auto& parameter : allParameters)
    {
        bool alreadyAdded = false;
        for (const auto& addedParameter : addedParameters)
        {
            if (addedParameter == parameter)
            {
                alreadyAdded = true;
                break;
            }
        }

        if (!alreadyAdded)
        {
            result.push_back(ParameterPair(parameter.getName(), parameter.getValues().at(0)));
            addedParameters.push_back(parameter);
        }
    }

    return result;
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

size_t ConfigurationManager::getConfigurationCountForParameters(const std::vector<KernelParameter>& parameters)
{
    size_t result = 1;

    for (const auto& parameter : parameters)
    {
        result *= parameter.getValues().size();
    }

    return result;
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
