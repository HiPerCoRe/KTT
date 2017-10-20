#include <fstream>
#include <sstream>
#include "kernel_manager.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelManager::KernelManager() :
    nextId(0),
    globalSizeType(GlobalSizeType::Opencl)
{}

KernelId KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    DimensionVector convertedGlobalSize = globalSize;

    if (globalSizeType == GlobalSizeType::Cuda)
    {
        convertedGlobalSize.multiply(localSize);
    }

    kernels.emplace_back(nextId, source, kernelName, convertedGlobalSize, localSize);
    return nextId++;
}

KernelId KernelManager::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filePath);
    return addKernel(source, kernelName, globalSize, localSize);
}

KernelId KernelManager::addKernelComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds)
{
    if (!containsUnique(kernelIds))
    {
        throw std::runtime_error("Kernels added to kernel composition must be unique");
    }

    std::vector<const Kernel*> compositionKernels;
    for (const auto& id : kernelIds)
    {
        if (!isKernel(id))
        {
            throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
        }
        compositionKernels.push_back(&kernels.at(id));
    }

    kernelCompositions.emplace_back(nextId, compositionName, compositionKernels);
    return nextId++;
}

std::string KernelManager::getKernelSourceWithDefines(const KernelId id, const KernelConfiguration& configuration) const
{
    std::string source = getKernel(id).getSource();

    for (const auto& parameterPair : configuration.getParameterPairs())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterPair); // clean way to convert number to string
        source = std::string("#define ") + std::get<0>(parameterPair) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
}

KernelConfiguration KernelManager::getKernelConfiguration(const KernelId id, const std::vector<ParameterPair>& parameterPairs) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = getKernel(id);
    DimensionVector global = kernel.getGlobalSize();
    DimensionVector local = kernel.getLocalSize();
    
    for (const auto& parameterPair : parameterPairs)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernel.getParameters())
        {
            if (parameter.getName() == std::get<0>(parameterPair))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <") + std::get<0>(parameterPair) + "> is not associated with kernel with id: "
                + std::to_string(id));
        }

        modifyDimensionVector(global, DimensionVectorType::Global, *targetParameter, std::get<1>(parameterPair));
        modifyDimensionVector(local, DimensionVectorType::Local, *targetParameter, std::get<1>(parameterPair));
    }

    return KernelConfiguration(global, local, parameterPairs, globalSizeType);
}

KernelConfiguration KernelManager::getKernelCompositionConfiguration(const KernelId compositionId,
    const std::vector<ParameterPair>& parameterPairs) const
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    const KernelComposition& kernelComposition = getKernelComposition(compositionId);
    std::vector<std::pair<KernelId, DimensionVector>> globalSizes;
    std::vector<std::pair<KernelId, DimensionVector>> localSizes;

    for (const auto& kernel : kernelComposition.getKernels())
    {
        globalSizes.push_back(std::make_pair(kernel->getId(), kernel->getGlobalSize()));
        localSizes.push_back(std::make_pair(kernel->getId(), kernel->getLocalSize()));
    }
    
    for (const auto& parameterPair : parameterPairs)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernelComposition.getParameters())
        {
            if (parameter.getName() == std::get<0>(parameterPair))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <") + std::get<0>(parameterPair)
                + "> is not associated with kernel composition with id: " + std::to_string(compositionId));
        }

        for (auto& globalSizePair : globalSizes)
        {
            for (const auto kernelId : targetParameter->getCompositionKernels())
            {
                if (globalSizePair.first == kernelId)
                {
                    modifyDimensionVector(globalSizePair.second, DimensionVectorType::Global, *targetParameter, std::get<1>(parameterPair));
                }
            }
        }
        for (auto& localSizePair : localSizes)
        {
            for (const auto kernelId : targetParameter->getCompositionKernels())
            {
                if (localSizePair.first == kernelId)
                {
                    modifyDimensionVector(localSizePair.second, DimensionVectorType::Local, *targetParameter, std::get<1>(parameterPair));
                }
            }
        }
    }

    return KernelConfiguration(globalSizes, localSizes, parameterPairs, globalSizeType);
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurations(const KernelId id, const DeviceInfo& deviceInfo) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<KernelConfiguration> configurations;
    const Kernel& kernel = getKernel(id);

    if (kernel.getParameters().size() == 0)
    {
        configurations.emplace_back(kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<ParameterPair>{}, globalSizeType);
    }
    else
    {
        computeConfigurations(0, deviceInfo, kernel.getParameters(), kernel.getConstraints(), std::vector<ParameterPair>(0), kernel.getGlobalSize(),
            kernel.getLocalSize(), kernel.getLocalSize(), configurations);
    }
    return configurations;
}

std::vector<KernelConfiguration> KernelManager::getKernelCompositionConfigurations(const KernelId compositionId, const DeviceInfo& deviceInfo) const
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    const KernelComposition& composition = getKernelComposition(compositionId);
    std::vector<std::pair<KernelId, DimensionVector>> globalSizes;
    std::vector<std::pair<KernelId, DimensionVector>> localSizes;

    for (const auto& kernel : composition.getKernels())
    {
        globalSizes.push_back(std::make_pair(kernel->getId(), kernel->getGlobalSize()));
        localSizes.push_back(std::make_pair(kernel->getId(), kernel->getLocalSize()));
    }

    std::vector<KernelConfiguration> kernelConfigurations;
    if (composition.getParameters().size() == 0)
    {
        kernelConfigurations.emplace_back(globalSizes, localSizes, std::vector<ParameterPair>{}, globalSizeType);
    }
    else
    {
        computeCompositionConfigurations(0, deviceInfo, composition.getParameters(), composition.getConstraints(), std::vector<ParameterPair>(0),
            globalSizes, localSizes, localSizes, kernelConfigurations);
    }

    return kernelConfigurations;
}

void KernelManager::setGlobalSizeType(const GlobalSizeType& type)
{
    this->globalSizeType = type;
}

void KernelManager::addParameter(const KernelId id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension)
{
    if (isKernel(id))
    {
        getKernel(id).addParameter(KernelParameter(name, values, modifierType, modifierAction, modifierDimension));
    }
    else if (isComposition(id))
    {
        getKernelComposition(id).addParameter(KernelParameter(name, values, modifierType, modifierAction, modifierDimension));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    if (isKernel(id))
    {
        getKernel(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else if (isComposition(id))
    {
        getKernelComposition(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::setArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    if (isKernel(id))
    {
        getKernel(id).setArguments(argumentIds);
    }
    else if (isComposition(id))
    {
        getKernelComposition(id).setSharedArguments(argumentIds);
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::setTuningManipulatorFlag(const KernelId id, const TunerFlag flag)
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    getKernel(id).setTuningManipulatorFlag(flag);
}

void KernelManager::addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
    const Dimension& modifierDimension)
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(kernelId).addKernelParameter(kernelId, KernelParameter(parameterName, parameterValues, modifierType, modifierAction,
        modifierDimension));
}

void KernelManager::setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(compositionId).setKernelArguments(kernelId, argumentIds);
}

const Kernel& KernelManager::getKernel(const KernelId id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return kernel;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

Kernel& KernelManager::getKernel(const KernelId id)
{
    return const_cast<Kernel&>(static_cast<const KernelManager*>(this)->getKernel(id));
}

size_t KernelManager::getCompositionCount() const
{
    return kernelCompositions.size();
}

const KernelComposition& KernelManager::getKernelComposition(const KernelId id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return kernelComposition;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
}

KernelComposition& KernelManager::getKernelComposition(const KernelId id)
{
    return const_cast<KernelComposition&>(static_cast<const KernelManager*>(this)->getKernelComposition(id));
}

bool KernelManager::isKernel(const KernelId id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return true;
        }
    }

    return false;
}

bool KernelManager::isComposition(const KernelId id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return true;
        }
    }

    return false;
}

std::string KernelManager::loadFileToString(const std::string& filePath) const
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

void KernelManager::computeConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterPair>& parameterPairs, const DimensionVector& globalSize, const DimensionVector& localSize,
    const DimensionVector& defaultLocalSize, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterPairs, globalSizeType);
        if (configurationIsValid(configuration, constraints, deviceInfo))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterPairs = parameterPairs;
        newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

        DimensionVector newGlobalSize = globalSize;
        DimensionVector newLocalSize = localSize;

        if (globalSizeType == GlobalSizeType::Cuda)
        {
            newGlobalSize.divide(defaultLocalSize);
        }

        modifyDimensionVector(newGlobalSize, DimensionVectorType::Global, parameter, value);
        modifyDimensionVector(newLocalSize, DimensionVectorType::Local, parameter, value);

        if (globalSizeType == GlobalSizeType::Cuda)
        {
            newGlobalSize.multiply(defaultLocalSize);
        }

        computeConfigurations(currentParameterIndex + 1, deviceInfo, parameters, constraints, newParameterPairs, newGlobalSize, newLocalSize,
            defaultLocalSize, finalResult);
    }
}

void KernelManager::computeCompositionConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterPair>& parameterPairs, std::vector<std::pair<KernelId, DimensionVector>>& globalSizes,
    std::vector<std::pair<KernelId, DimensionVector>>& localSizes, std::vector<std::pair<KernelId, DimensionVector>>& defaultLocalSizes,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSizes, localSizes, parameterPairs, globalSizeType);
        if (configurationIsValid(configuration, constraints, deviceInfo))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterPairs = parameterPairs;
        newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

        for (const auto compositionKernelId : parameter.getCompositionKernels())
        {
            for (auto& globalSizePair : globalSizes)
            {
                if (compositionKernelId == globalSizePair.first)
                {
                    if (globalSizeType == GlobalSizeType::Cuda)
                    {
                        for (const auto& defaultLocalSizePair : defaultLocalSizes)
                        {
                            if (globalSizePair.first == defaultLocalSizePair.first)
                            {
                                globalSizePair.second.divide(defaultLocalSizePair.second);
                                break;
                            }
                        }
                    }
                    modifyDimensionVector(globalSizePair.second, DimensionVectorType::Global, parameter, value);
                    if (globalSizeType == GlobalSizeType::Cuda)
                    {
                        for (const auto& defaultLocalSizePair : defaultLocalSizes)
                        {
                            if (globalSizePair.first == defaultLocalSizePair.first)
                            {
                                globalSizePair.second.multiply(defaultLocalSizePair.second);
                                break;
                            }
                        }
                    }
                }
            }

            for (auto& localSizePair : localSizes)
            {
                if (compositionKernelId == localSizePair.first)
                {
                    modifyDimensionVector(localSizePair.second, DimensionVectorType::Local, parameter, value);
                }
            }
        }

        computeCompositionConfigurations(currentParameterIndex + 1, deviceInfo, parameters, constraints, newParameterPairs, globalSizes, localSizes,
            defaultLocalSizes, finalResult);
    }
}

void KernelManager::modifyDimensionVector(DimensionVector& vector, const DimensionVectorType& dimensionVectorType, const KernelParameter& parameter,
    const size_t parameterValue) const
{
    if (parameter.getModifierType() == ThreadModifierType::None
        || dimensionVectorType == DimensionVectorType::Global && parameter.getModifierType() == ThreadModifierType::Local
        || dimensionVectorType == DimensionVectorType::Local && parameter.getModifierType() == ThreadModifierType::Global)
    {
        return;
    }

    vector.modifyByValue(parameterValue, parameter.getModifierAction(), parameter.getModifierDimension());
}

bool KernelManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints,
    const DeviceInfo& deviceInfo) const
{
    for (const auto& constraint : constraints)
    {
        std::vector<std::string> constraintNames = constraint.getParameterNames();
        auto constraintValues = std::vector<size_t>(constraintNames.size());

        for (size_t i = 0; i < constraintNames.size(); i++)
        {
            for (const auto& parameterPair : configuration.getParameterPairs())
            {
                if (std::get<0>(parameterPair) == constraintNames.at(i))
                {
                    constraintValues.at(i) = std::get<1>(parameterPair);
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

} // namespace ktt
