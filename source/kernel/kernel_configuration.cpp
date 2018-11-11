#include <kernel/kernel_configuration.h>

namespace ktt
{

KernelConfiguration::KernelConfiguration() :
    compositeConfiguration(false),
    validConfiguration(false)
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
    compositeConfiguration(false),
    validConfiguration(true)
{}

KernelConfiguration::KernelConfiguration(const std::map<KernelId, DimensionVector>& compositionGlobalSizes,
    const std::map<KernelId, DimensionVector>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs) :
    KernelConfiguration(compositionGlobalSizes, compositionLocalSizes, parameterPairs, std::map<KernelId, std::vector<LocalMemoryModifier>>{})
{}

KernelConfiguration::KernelConfiguration(const std::map<KernelId, DimensionVector>& compositionGlobalSizes,
    const std::map<KernelId, DimensionVector>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs,
    const std::map<KernelId, std::vector<LocalMemoryModifier>>& compositionLocalMemoryModifiers) :
    globalSize(DimensionVector()),
    localSize(DimensionVector()),
    compositionGlobalSizes(compositionGlobalSizes),
    compositionLocalSizes(compositionLocalSizes),
    compositionLocalMemoryModifiers(compositionLocalMemoryModifiers),
    parameterPairs(parameterPairs),
    compositeConfiguration(true),
    validConfiguration(true)
{}

const DimensionVector& KernelConfiguration::getGlobalSize() const
{
    return globalSize;
}

const DimensionVector& KernelConfiguration::getLocalSize() const
{
    return localSize;
}

const std::vector<LocalMemoryModifier>& KernelConfiguration::getLocalMemoryModifiers() const
{
    return localMemoryModifiers;
}

const DimensionVector& KernelConfiguration::getCompositionKernelGlobalSize(const KernelId id) const
{
    if (compositionGlobalSizes.find(id) == compositionGlobalSizes.end())
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    return compositionGlobalSizes.find(id)->second;
}

const DimensionVector& KernelConfiguration::getCompositionKernelLocalSize(const KernelId id) const
{
    if (compositionLocalSizes.find(id) == compositionLocalSizes.end())
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    return compositionLocalSizes.find(id)->second;
}

std::vector<LocalMemoryModifier> KernelConfiguration::getCompositionKernelLocalMemoryModifiers(const KernelId id) const
{
    if (compositionLocalMemoryModifiers.find(id) == compositionLocalMemoryModifiers.end())
    {
        return std::vector<LocalMemoryModifier>{};
    }

    return compositionLocalMemoryModifiers.find(id)->second;
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

const std::vector<ParameterPair>& KernelConfiguration::getParameterPairs() const
{
    return parameterPairs;
}

bool KernelConfiguration::isComposite() const
{
    return compositeConfiguration;
}

bool KernelConfiguration::isValid() const
{
    return validConfiguration;
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
            outputTarget << "global size " << i << " " << globalSize << ", ";
            outputTarget << "local size " << i << " " << localSize << ", ";
        }
        else
        {
            outputTarget << "global size " << globalSize << ", ";
            outputTarget << "local size " << localSize << ", ";
        }
    }

    outputTarget << "parameters: ";
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
