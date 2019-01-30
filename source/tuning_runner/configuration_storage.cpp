#include <stdexcept>
#include <tuning_runner/configuration_storage.h>
#include <utility/ktt_utility.h>

namespace ktt
{

void ConfigurationStorage::storeConfiguration(const std::pair<KernelConfiguration, uint64_t>& configuration)
{
    orderedConfigurations.insert(std::make_pair(configuration.second, configuration.first));
}

void ConfigurationStorage::storeProcessedPack(const KernelParameterPack& pack)
{
    processedPacks.push_back(pack);
}

KernelConfiguration ConfigurationStorage::getBestCompatibleConfiguration(const KernelParameterPack& currentPack,
    const std::vector<ParameterPair>& generatedPairs) const
{
    for (const auto& configuration : orderedConfigurations)
    {
        if (isConfigurationCompatible(configuration.second, currentPack, generatedPairs))
        {
            return configuration.second;
        }
    }

    return KernelConfiguration();
}

bool ConfigurationStorage::isConfigurationCompatible(const KernelConfiguration& configuration, const KernelParameterPack& currentPack,
    const std::vector<ParameterPair>& generatedPairs) const
{
    std::vector<ParameterPair> sharedPairs;

    for (const auto& pair : generatedPairs)
    {
        if (!elementExists(pair.getName(), currentPack.getParameterNames()))
        {
            continue;
        }

        bool sharedPair = false;

        for (const auto& processedPack : processedPacks)
        {
            if (elementExists(pair.getName(), processedPack.getParameterNames()))
            {
                sharedPair = true;
                break;
            }
        }

        if (sharedPair)
        {
            sharedPairs.push_back(pair);
        }
    }

    for (const auto& pair : sharedPairs)
    {
        for (const auto& configurationPair : configuration.getParameterPairs())
        {
            if (pair.getName() == configurationPair.getName())
            {
                if (pair.hasValueDouble() && !floatEquals(pair.getValueDouble(), configurationPair.getValueDouble())
                    || !pair.hasValueDouble() && pair.getValue() != configurationPair.getValue())
                {
                    return false;
                }
            }
        }
    }
    
    return true;
}

} // namespace ktt
