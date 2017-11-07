#include "kernel_configuration.h"

namespace ktt
{

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs) :
    globalSize(globalSize),
    localSize(localSize),
    parameterPairs(parameterPairs),
    compositeConfiguration(false)
{}
    
KernelConfiguration::KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
    const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs) :
    globalSize(DimensionVector()),
    localSize(DimensionVector()),
    compositionGlobalSizes(compositionGlobalSizes),
    compositionLocalSizes(compositionLocalSizes),
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
        outputTarget << std::get<0>(parameterPair) << ": " << std::get<1>(parameterPair) << " ";
    }

    return outputTarget;
}

} // namespace ktt
