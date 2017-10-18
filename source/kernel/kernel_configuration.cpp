#include "kernel_configuration.h"

namespace ktt
{

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterValue>& parameterValues, const GlobalSizeType& globalSizeType) :
    globalSize(globalSize),
    localSize(localSize),
    parameterValues(parameterValues),
    globalSizeType(globalSizeType),
    compositeConfiguration(false)
{}
    
KernelConfiguration::KernelConfiguration(const std::vector<std::pair<size_t, DimensionVector>>& globalSizes,
    const std::vector<std::pair<size_t, DimensionVector>>& localSizes, const std::vector<ParameterValue>& parameterValues,
    const GlobalSizeType& globalSizeType) :
    globalSize(DimensionVector(1, 1, 1)),
    localSize(DimensionVector(1, 1, 1)),
    globalSizes(globalSizes),
    localSizes(localSizes),
    parameterValues(parameterValues),
    globalSizeType(globalSizeType),
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

DimensionVector KernelConfiguration::getGlobalSize(const size_t kernelId) const
{
    for (const auto& element : globalSizes)
    {
        if (element.first == kernelId)
        {
            return element.second;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
}

DimensionVector KernelConfiguration::getLocalSize(const size_t kernelId) const
{
    for (const auto& element : localSizes)
    {
        if (element.first == kernelId)
        {
            return element.second;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
}

std::vector<DimensionVector> KernelConfiguration::getGlobalSizes() const
{
    if (globalSizes.size() > 0)
    {
        std::vector<DimensionVector> globalSizesResult;

        for (const auto& globalSizePair : globalSizes)
        {
            globalSizesResult.push_back(globalSizePair.second);
        }

        return globalSizesResult;
    }

    return std::vector<DimensionVector>{globalSize};
}

std::vector<DimensionVector> KernelConfiguration::getLocalSizes() const
{
    if (localSizes.size() > 0)
    {
        std::vector<DimensionVector> localSizesResult;

        for (const auto& localSizePair : localSizes)
        {
            localSizesResult.push_back(localSizePair.second);
        }

        return localSizesResult;
    }

    return std::vector<DimensionVector>{localSize};
}

std::vector<ParameterValue> KernelConfiguration::getParameterValues() const
{
    return parameterValues;
}

GlobalSizeType KernelConfiguration::getGlobalSizeType() const
{
    return globalSizeType;
}

bool KernelConfiguration::isComposite() const
{
    return compositeConfiguration;
}

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration)
{
    std::vector<DimensionVector> globalSizes = kernelConfiguration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = kernelConfiguration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector convertedGlobalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        if (kernelConfiguration.globalSizeType == GlobalSizeType::Cuda)
        {
            std::get<0>(convertedGlobalSize) = std::get<0>(convertedGlobalSize) / std::get<0>(localSize);
            std::get<1>(convertedGlobalSize) = std::get<1>(convertedGlobalSize) / std::get<1>(localSize);
            std::get<2>(convertedGlobalSize) = std::get<2>(convertedGlobalSize) / std::get<2>(localSize);
        }

        if (globalSizes.size() > 1)
        {
            outputTarget << "global size " << i << ": " << std::get<0>(convertedGlobalSize) << ", " << std::get<1>(convertedGlobalSize) << ", "
                << std::get<2>(convertedGlobalSize) << "; ";
            outputTarget << "local size " << i << ": " << std::get<0>(localSize) << ", " << std::get<1>(localSize) << ", " << std::get<2>(localSize)
                << "; ";
        }
        else
        {
            outputTarget << "global size: " << std::get<0>(convertedGlobalSize) << ", " << std::get<1>(convertedGlobalSize) << ", "
                << std::get<2>(convertedGlobalSize) << "; ";
            outputTarget << "local size: " << std::get<0>(localSize) << ", " << std::get<1>(localSize) << ", " << std::get<2>(localSize) << "; ";
        }
    }
    
    outputTarget << "parameters: ";
    if (kernelConfiguration.parameterValues.size() == 0)
    {
        outputTarget << "none";
    }
    for (const auto& value : kernelConfiguration.parameterValues)
    {
        outputTarget << std::get<0>(value) << ": " << std::get<1>(value) << " ";
    }
    outputTarget << std::endl;

    return outputTarget;
}

} // namespace ktt
