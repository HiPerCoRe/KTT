#include <fstream>
#include <sstream>
#include "kernel_manager.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelManager::KernelManager(const DeviceInfo& currentDeviceInfo) :
    nextId(0),
    currentDeviceInfo(currentDeviceInfo)
{}

KernelId KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    kernels.emplace_back(nextId, source, kernelName, globalSize, localSize);
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
    return getKernelSourceWithDefines(id, configuration.getParameterPairs());
}

std::string KernelManager::getKernelSourceWithDefines(const KernelId id, const std::vector<ParameterPair>& configuration) const
{
    std::string source = getKernel(id).getSource();

    for (const auto& parameterPair : configuration)
    {
        std::stringstream stream;
        if (!parameterPair.hasValueDouble())
        {
            stream << parameterPair.getValue();
        }
        else
        {
            stream << parameterPair.getValueDouble();
        }
        source = std::string("#define ") + parameterPair.getName() + std::string(" ") + stream.str() + std::string("\n") + source;
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
    std::vector<LocalMemoryModifier> modifiers;

    for (const auto& parameterPair : parameterPairs)
    {
        bool parameterFound = false;

        for (const auto& parameter : kernel.getParameters())
        {
            if (parameter.getName() == parameterPair.getName())
            {
                if (parameter.getModifierType() == ModifierType::Global)
                {
                    global.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(), parameter.getModifierDimension());
                }
                else if (parameter.getModifierType() == ModifierType::Local)
                {
                    local.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(), parameter.getModifierDimension());
                }
                else if (parameter.getModifierType() == ModifierType::Both) {
                    local.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(), parameter.getModifierDimension());
                    global.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(), parameter.getModifierDimension());
                }

                if (parameter.isLocalMemoryModifier())
                {
                    std::vector<std::pair<ArgumentId, ModifierAction>> localArguments = parameter.getLocalMemoryArguments();
                    for (const auto& argument : localArguments)
                    {
                        modifiers.emplace_back(id, argument.first, argument.second, parameterPair.getValue());
                    }
                }

                parameterFound = true;
                break;
            }
        }

        if (!parameterFound)
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName() + " is not associated with kernel with id: "
                + std::to_string(id));
        }
    }

    return KernelConfiguration(global, local, parameterPairs, modifiers);
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
    std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>> modifiers;

    for (const auto& kernel : kernelComposition.getKernels())
    {
        globalSizes.push_back(std::make_pair(kernel->getId(), kernel->getGlobalSize()));
        localSizes.push_back(std::make_pair(kernel->getId(), kernel->getLocalSize()));
    }
    
    for (const auto& parameterPair : parameterPairs)
    {
        bool parameterFound = false;

        for (const auto& parameter : kernelComposition.getParameters())
        {
            if (parameter.getName() == parameterPair.getName())
            {
                for (auto& globalSizePair : globalSizes)
                {
                    for (const auto kernelId : parameter.getCompositionKernels())
                    {
                        if (globalSizePair.first == kernelId && (parameter.getModifierType() == ModifierType::Global || parameter.getModifierType() == ModifierType::Both))
                        {
                            globalSizePair.second.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(),
                                parameter.getModifierDimension());
                        }
                    }
                }
                for (auto& localSizePair : localSizes)
                {
                    for (const auto kernelId : parameter.getCompositionKernels())
                    {
                        if (localSizePair.first == kernelId && (parameter.getModifierType() == ModifierType::Local || parameter.getModifierType() == ModifierType::Both))
                        {
                            localSizePair.second.modifyByValue(parameterPair.getValue(), parameter.getModifierAction(),
                                parameter.getModifierDimension());
                        }
                    }
                }

                if (parameter.isLocalMemoryModifier())
                {
                    std::vector<size_t> kernels = parameter.getLocalMemoryModifierKernels();
                    for (const auto& kernel : kernels)
                    {
                        std::vector<std::pair<ArgumentId, ModifierAction>> localArguments = parameter.getLocalMemoryArguments();
                        std::vector<LocalMemoryModifier> currentModifiers;

                        for (const auto& argument : localArguments)
                        {
                            currentModifiers.emplace_back(kernel, argument.first, argument.second, parameterPair.getValue());
                        }

                        modifiers.push_back(std::make_pair(kernel, currentModifiers));
                    }
                }

                parameterFound = true;
                break;
            }
        }

        if (!parameterFound)
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName()
                + " is not associated with kernel composition with id: " + std::to_string(compositionId));
        }
    }

    return KernelConfiguration(globalSizes, localSizes, parameterPairs, modifiers);
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurations(const KernelId id) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<KernelConfiguration> configurations;
    const Kernel& kernel = getKernel(id);

    if (kernel.getParameters().size() == 0)
    {
        configurations.emplace_back(kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<ParameterPair>{});
    }
    else
    {
        computeConfigurations(kernel.getId(), 0, kernel.getParameters(), kernel.getConstraints(), std::vector<ParameterPair>{},
            kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<LocalMemoryModifier>{}, configurations);
    }
    return configurations;
}

std::vector<KernelConfiguration> KernelManager::getKernelCompositionConfigurations(const KernelId compositionId) const
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
        kernelConfigurations.emplace_back(globalSizes, localSizes, std::vector<ParameterPair>{});
    }
    else
    {
        computeCompositionConfigurations(0, composition.getParameters(), composition.getConstraints(), std::vector<ParameterPair>{}, globalSizes,
            localSizes, std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>{}, kernelConfigurations);
    }

    return kernelConfigurations;
}

void KernelManager::addParameter(const KernelId id, const std::string& name, const std::vector<size_t>& values, const ModifierType modifierType,
    const ModifierAction modifierAction, const ModifierDimension modifierDimension)
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

void KernelManager::addParameter(const KernelId id, const std::string& name, const std::vector<double>& values)
{
    if (isKernel(id))
    {
        getKernel(id).addParameter(KernelParameter(name, values));
    }
    else if (isComposition(id))
    {
        getKernelComposition(id).addParameter(KernelParameter(name, values));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    throw std::runtime_error("Unsupported operation");
}

void KernelManager::addLocalMemoryModifier(const KernelId id, const std::string& parameterName, const ArgumentId argumentId,
    const ModifierAction modifierAction)
{
    if (isKernel(id))
    {
        getKernel(id).addLocalMemoryModifier(parameterName, argumentId, modifierAction);
    }
    else if (isComposition(id))
    {
        getKernelComposition(id).addLocalMemoryModifier(parameterName, argumentId, modifierAction);
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

void KernelManager::setTuningManipulatorFlag(const KernelId id, const bool flag)
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    getKernel(id).setTuningManipulatorFlag(flag);
}

void KernelManager::addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ModifierType modifierType, const ModifierAction modifierAction,
    const ModifierDimension modifierDimension)
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(compositionId).addKernelParameter(kernelId, KernelParameter(parameterName, parameterValues, modifierType, modifierAction,
        modifierDimension));
}

void KernelManager::addCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
    const ArgumentId argumentId, const ModifierAction modifierAction)
{
    if (!isComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(compositionId).addKernelLocalMemoryModifier(kernelId, parameterName, argumentId, modifierAction);
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

std::string KernelManager::loadFileToString(const std::string& filePath)
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

void KernelManager::computeConfigurations(const KernelId kernelId, const size_t currentParameterIndex,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterPair>& parameterPairs, const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<LocalMemoryModifier>& modifiers, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterPairs, modifiers);
        if (configurationIsValid(configuration, constraints))
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
            auto newParameterPairs = parameterPairs;
            newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

            DimensionVector newGlobalSize = globalSize;
            DimensionVector newLocalSize = localSize;

            if (parameter.getModifierType() == ModifierType::Global)
            {
                newGlobalSize.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
            }
            else if (parameter.getModifierType() == ModifierType::Local)
            {
                newLocalSize.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
            }
            else if (parameter.getModifierType() == ModifierType::Both)
            {
                newGlobalSize.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
                newLocalSize.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
            }

            std::vector<LocalMemoryModifier> newModifiers = modifiers;
            if (parameter.isLocalMemoryModifier())
            {
                std::vector<std::pair<ArgumentId, ModifierAction>> localArguments = parameter.getLocalMemoryArguments();
                for (const auto& argument : localArguments)
                {
                    newModifiers.emplace_back(kernelId, argument.first, argument.second, value);
                }
            }

            computeConfigurations(kernelId, currentParameterIndex + 1, parameters, constraints, newParameterPairs, newGlobalSize, newLocalSize,
                newModifiers, finalResult);
        }
    }
    else
    {
        for (const auto& value : parameter.getValuesDouble()) // recursively build tree of configurations for each parameter value
        {
            auto newParameterPairs = parameterPairs;
            newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

            DimensionVector newGlobalSize = globalSize;
            DimensionVector newLocalSize = localSize;
            std::vector<LocalMemoryModifier> newModifiers = modifiers;

            computeConfigurations(kernelId, currentParameterIndex + 1, parameters, constraints, newParameterPairs, newGlobalSize, newLocalSize,
                newModifiers, finalResult);
        }
    }
}

void KernelManager::computeCompositionConfigurations(const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
    const std::vector<KernelConstraint>& constraints, const std::vector<ParameterPair>& parameterPairs,
    const std::vector<std::pair<KernelId, DimensionVector>>& globalSizes, const std::vector<std::pair<KernelId, DimensionVector>>& localSizes,
    const std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>& modifiers, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSizes, localSizes, parameterPairs, modifiers);
        if (configurationIsValid(configuration, constraints))
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
            auto newParameterPairs = parameterPairs;
            newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

            std::vector<std::pair<KernelId, DimensionVector>> newGlobalSizes = globalSizes;
            std::vector<std::pair<KernelId, DimensionVector>> newLocalSizes = localSizes;

            for (const auto compositionKernelId : parameter.getCompositionKernels())
            {
                for (auto& globalSizePair : newGlobalSizes)
                {
                    if (compositionKernelId == globalSizePair.first && (parameter.getModifierType() == ModifierType::Global || parameter.getModifierType() == ModifierType::Both))
                    {
                        globalSizePair.second.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
                    }
                }

                for (auto& localSizePair : newLocalSizes)
                {
                    if (compositionKernelId == localSizePair.first && (parameter.getModifierType() == ModifierType::Local || parameter.getModifierType() == ModifierType::Both))
                    {
                        localSizePair.second.modifyByValue(value, parameter.getModifierAction(), parameter.getModifierDimension());
                    }
                }
            }

            std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>> newModifiers = modifiers;
            if (parameter.isLocalMemoryModifier())
            {
                std::vector<size_t> kernels = parameter.getLocalMemoryModifierKernels();
                for (const auto& kernel : kernels)
                {
                    std::vector<std::pair<ArgumentId, ModifierAction>> localArguments = parameter.getLocalMemoryArguments();
                    std::vector<LocalMemoryModifier> currentModifiers;

                    for (const auto& argument : localArguments)
                    {
                        currentModifiers.emplace_back(kernel, argument.first, argument.second, value);
                    }

                    newModifiers.push_back(std::make_pair(kernel, currentModifiers));
                }
            }

            computeCompositionConfigurations(currentParameterIndex + 1, parameters, constraints, newParameterPairs, newGlobalSizes, newLocalSizes,
                newModifiers, finalResult);
        }
    }
    else
    {
        for (const auto& value : parameter.getValuesDouble()) // recursively build tree of configurations for each parameter value
        {
            auto newParameterPairs = parameterPairs;
            newParameterPairs.push_back(ParameterPair(parameter.getName(), value));

            std::vector<std::pair<KernelId, DimensionVector>> newGlobalSizes = globalSizes;
            std::vector<std::pair<KernelId, DimensionVector>> newLocalSizes = localSizes;
            std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>> newModifiers = modifiers;

            computeCompositionConfigurations(currentParameterIndex + 1, parameters, constraints, newParameterPairs, newGlobalSizes, newLocalSizes,
                newModifiers, finalResult);
        }
    }
}

bool KernelManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const
{
    for (const auto& constraint : constraints)
    {
        std::vector<std::string> constraintNames = constraint.getParameterNames();
        auto constraintValues = std::vector<size_t>(constraintNames.size());

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
        if (localSize.getTotalSize() > currentDeviceInfo.getMaxWorkGroupSize())
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt
