#include "kernel_configuration.h"

namespace ktt
{

KernelConfiguration::KernelConfiguration() :
    compositeConfiguration(false)
{}

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs) :
    KernelConfiguration(globalSize, localSize, parameterPairs, std::vector<std::tuple<ArgumentId, ModifierAction, size_t>>{})
{}

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs,
    const std::vector<std::tuple<ArgumentId, ModifierAction, size_t>>& localMemoryArgumentModifiers) :
    globalSize(globalSize),
    localSize(localSize),
    localMemoryArgumentModifiers(localMemoryArgumentModifiers),
    parameterPairs(parameterPairs),
    compositeConfiguration(false)
{}

KernelConfiguration::KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
    const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs) :
    KernelConfiguration(compositionGlobalSizes, compositionLocalSizes, parameterPairs,
        std::vector<std::pair<KernelId, std::vector<std::tuple<ArgumentId, ModifierAction, size_t>>>>{})
{}

KernelConfiguration::KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
    const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs,
    const std::vector<std::pair<KernelId, std::vector<std::tuple<ArgumentId, ModifierAction, size_t>>>>& compositionLocalMemoryArgumentModifiers) :
    globalSize(DimensionVector()),
    localSize(DimensionVector()),
    compositionGlobalSizes(compositionGlobalSizes),
    compositionLocalSizes(compositionLocalSizes),
    compositionLocalMemoryArgumentModifiers(compositionLocalMemoryArgumentModifiers),
    parameterPairs(parameterPairs),
    compositeConfiguration(true)
{}

DimensionVector KernelConfiguration::getGlobalSize() const
{
    return globalSize;
}

DimensionVector KernelConfiguration::getLocalSize() const
{
    return localSize;
}

std::vector<std::tuple<ArgumentId, ModifierAction, size_t>> KernelConfiguration::getLocalMemoryArgumentModifiers() const
{
    return localMemoryArgumentModifiers;
}

DimensionVector KernelConfiguration::getCompositionKernelGlobalSize(const KernelId id) const
{
    for (const auto& globalSizePair : compositionGlobalSizes)
    {
        if (globalSizePair.first == id)
        {
            return globalSizePair.second;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

DimensionVector KernelConfiguration::getCompositionKernelLocalSize(const KernelId id) const
{
    for (const auto& localSizePair : compositionLocalSizes)
    {
        if (localSizePair.first == id)
        {
            return localSizePair.second;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

std::vector<std::tuple<ArgumentId, ModifierAction, size_t>> KernelConfiguration::getCompositionKernelLocalMemoryArgumentModifiers(
    const KernelId id) const
{
    for (const auto& modifier : compositionLocalMemoryArgumentModifiers)
    {
        if (modifier.first == id)
        {
            return modifier.second;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

std::vector<DimensionVector> KernelConfiguration::getGlobalSizes() const
{
    if (compositionGlobalSizes.size() > 0)
    {
        std::vector<DimensionVector> globalSizes;

        for (const auto& globalSizePair : compositionGlobalSizes)
        {
            globalSizes.push_back(globalSizePair.second);
        }

        return globalSizes;
    }

    return std::vector<DimensionVector>{globalSize};
}

std::vector<DimensionVector> KernelConfiguration::getLocalSizes() const
{
    if (compositionLocalSizes.size() > 0)
    {
        std::vector<DimensionVector> localSizes;

        for (const auto& localSizePair : compositionLocalSizes)
        {
            localSizes.push_back(localSizePair.second);
        }

        return localSizes;
    }

    return std::vector<DimensionVector>{localSize};
}

std::vector<ParameterPair> KernelConfiguration::getParameterPairs() const
{
    return parameterPairs;
}

bool KernelConfiguration::isComposite() const
{
    return compositeConfiguration;
}

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& configuration)
{
    std::vector<DimensionVector> globalSizes = configuration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector globalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        if (globalSizes.size() > 1)
        {
            outputTarget << "global size " << i << ": " << globalSize << "; ";
            outputTarget << "local size " << i << ": " << localSize << "; ";
        }
        else
        {
            outputTarget << "global size: " << globalSize << "; ";
            outputTarget << "local size: " << localSize << "; ";
        }
    }
    
    outputTarget << "parameters: ";
    if (configuration.parameterPairs.size() == 0)
    {
        outputTarget << "none";
    }
    for (const auto& parameterPair : configuration.parameterPairs)
    {
        outputTarget << parameterPair << " ";
    }

    return outputTarget;
}

} // namespace ktt
