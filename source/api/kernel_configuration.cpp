#include <api/kernel_configuration.h>

namespace ktt
{

KernelConfiguration::KernelConfiguration() :
    validConfiguration(false)
{}

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterPair>& parameterPairs) :
    globalSize(globalSize),
    localSize(localSize),
    parameterPairs(parameterPairs),
    validConfiguration(true)
{}

KernelConfiguration::KernelConfiguration(const std::map<KernelId, DimensionVector>& compositionGlobalSizes,
    const std::map<KernelId, DimensionVector>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs) :
    globalSize(DimensionVector()),
    localSize(DimensionVector()),
    compositionGlobalSizes(compositionGlobalSizes),
    compositionLocalSizes(compositionLocalSizes),
    parameterPairs(parameterPairs),
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
    if (configuration.getParameterPairs().empty())
    {
        outputTarget << "none";
    }

    const std::vector<ParameterPair>& parameterPairs = configuration.getParameterPairs();
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
