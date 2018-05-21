#include "kernel_configuration.h"

namespace ktt
{

KernelConfiguration::KernelConfiguration() :
    compositeConfiguration(false)
{}

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs) :
    KernelConfiguration(globalSize, localSize, parameterPairs, std::vector<LocalMemoryModifier>{})
{}

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs, const std::vector<LocalMemoryModifier>& localMemoryModifiers) :
    globalSize(globalSize),
    localSize(localSize),
    localMemoryModifiers(localMemoryModifiers),
    parameterPairs(parameterPairs),
    compositeConfiguration(false)
{}

KernelConfiguration::KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
    const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs) :
    KernelConfiguration(compositionGlobalSizes, compositionLocalSizes, parameterPairs,
        std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>{})
{}

KernelConfiguration::KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
    const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs,
    const std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>& compositionLocalMemoryModifiers) :
    globalSize(DimensionVector()),
    localSize(DimensionVector()),
    compositionGlobalSizes(compositionGlobalSizes),
    compositionLocalSizes(compositionLocalSizes),
    compositionLocalMemoryModifiers(compositionLocalMemoryModifiers),
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

std::vector<LocalMemoryModifier> KernelConfiguration::getLocalMemoryModifiers() const
{
    return localMemoryModifiers;
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

std::vector<LocalMemoryModifier> KernelConfiguration::getCompositionKernelLocalMemoryModifiers(const KernelId id) const
{
    for (const auto& modifier : compositionLocalMemoryModifiers)
    {
        if (modifier.first == id)
        {
            return modifier.second;
        }
    }

    return std::vector<LocalMemoryModifier>{};
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
    if (configuration.parameterPairs.size() == 0)
    {
        outputTarget << "none";
    }

    std::vector<ParameterPair> parameterPairs = configuration.parameterPairs;
    for (size_t i = 0; i < parameterPairs.size(); i++)
    {
        outputTarget << parameterPairs.at(i);
        if (i + 1 != parameterPairs.size())
        {
            outputTarget << ", ";
        }
    }

    return outputTarget;
}

} // namespace ktt
