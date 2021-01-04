#include <algorithm>
#include <stdexcept>
#include <kernel/kernel_composition.h>
#include <utility/ktt_utility.h>

namespace ktt
{

KernelComposition::KernelComposition(const KernelId id, const std::string& name, const std::vector<const Kernel*>& kernels) :
    id(id),
    name(name),
    kernels(kernels)
{
    for (const auto* kernel : kernels)
    {
        profiledKernels.insert(kernel->getId());
    }
}

void KernelComposition::addParameter(const KernelParameter& parameter)
{
    if (hasParameter(parameter.getName()))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }

    parameters.push_back(parameter);
}

void KernelComposition::addConstraint(const KernelConstraint& constraint)
{
    std::vector<std::string> parameterNames = constraint.getParameterNames();

    for (const auto& parameterName : parameterNames)
    {
        if (!hasParameter(parameterName))
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
        }
    }

    constraints.push_back(constraint);
}

void KernelComposition::addParameterPack(const KernelParameterPack& pack)
{
    for (const auto& existingPack : parameterPacks)
    {
        if (pack == existingPack)
        {
            throw std::runtime_error(std::string("The following parameter pack already exists: ") + pack.getName());
        }
    }

    if (!containsUnique(pack.getParameterNames()))
    {
        throw std::runtime_error(std::string("The following parameter pack contains duplicit parameter names: ") + pack.getName());
    }

    for (const auto& parameterName : pack.getParameterNames())
    {
        if (!hasParameter(parameterName))
        {
            throw std::runtime_error(std::string("Parameter with given name does not exist: ") + parameterName);
        }
    }
    parameterPacks.push_back(pack);
}

void KernelComposition::setSharedArguments(const std::vector<ArgumentId>& argumentIds)
{
    this->sharedArgumentIds = argumentIds;
}

void KernelComposition::setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    validateModifierParameters(parameterNames);

    switch (modifierType)
    {
    case ModifierType::Global:
        if (globalThreadModifiers.find(id) == globalThreadModifiers.end())
        {
            globalThreadModifiers.insert(std::make_pair(id,
                std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3>{nullptr, nullptr, nullptr}));
            globalThreadModifierNames.insert(std::make_pair(id, std::array<std::vector<std::string>, 3>{}));
        }
        globalThreadModifiers.find(id)->second[static_cast<size_t>(modifierDimension)] = modifierFunction;
        globalThreadModifierNames.find(id)->second[static_cast<size_t>(modifierDimension)] = parameterNames;
        break;
    case ModifierType::Local:
        if (localThreadModifiers.find(id) == localThreadModifiers.end())
        {
            localThreadModifiers.insert(std::make_pair(id,
                std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3>{nullptr, nullptr, nullptr}));
            localThreadModifierNames.insert(std::make_pair(id, std::array<std::vector<std::string>, 3>{}));
        }
        localThreadModifiers.find(id)->second[static_cast<size_t>(modifierDimension)] = modifierFunction;
        localThreadModifierNames.find(id)->second[static_cast<size_t>(modifierDimension)] = parameterNames;
        break;
    default:
        throw std::runtime_error("Unknown modifier type");
    }
}

void KernelComposition::setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)

{
    validateModifierParameters(parameterNames);

    if (localMemoryModifiers.find(id) == localMemoryModifiers.end())
    {
        localMemoryModifiers.insert(std::make_pair(id, std::map<ArgumentId, std::function<size_t(const size_t, const std::vector<size_t>&)>>{}));
        localMemoryModifierNames.insert(std::make_pair(id, std::map<ArgumentId, std::vector<std::string>>{}));
    }

    auto kernelModifierMap = localMemoryModifiers.find(id)->second;
    auto kernelModifierNameMap = localMemoryModifierNames.find(id)->second;
    if (kernelModifierMap.find(argumentId) != kernelModifierMap.end())
    {
        kernelModifierMap.erase(argumentId);
        kernelModifierNameMap.erase(argumentId);
    }
    kernelModifierMap.insert(std::make_pair(argumentId, modifierFunction));
    kernelModifierNameMap.insert(std::make_pair(argumentId, parameterNames));
}

void KernelComposition::setArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    if (kernelArgumentIds.find(id) != kernelArgumentIds.end())
    {
        kernelArgumentIds.erase(id);
    }
    kernelArgumentIds.insert(std::make_pair(id, argumentIds));
}

Kernel KernelComposition::transformToKernel() const
{
    Kernel kernel(id, "", name, DimensionVector(), DimensionVector());
    kernel.setTuningManipulatorFlag(true);

    for (const auto& parameter : parameters)
    {
        kernel.addParameter(parameter);
    }

    for (const auto& constraint : constraints)
    {
        kernel.addConstraint(constraint);
    }

    std::vector<ArgumentId> argumentIds;
    for (const auto id : sharedArgumentIds)
    {
        argumentIds.push_back(id);
    }

    for (const auto& kernel : kernels)
    {
        std::vector<ArgumentId> kernelArgumentIds = getKernelArgumentIds(kernel->getId());
        for (const auto id : kernelArgumentIds)
        {
            argumentIds.push_back(id);
        }
    }
    kernel.setArguments(argumentIds);

    return kernel;
}

std::map<KernelId, DimensionVector> KernelComposition::getModifiedGlobalSizes(const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName()
                + " is not associated with kernel composition with id " + std::to_string(id));
        }
    }

    std::map<KernelId, DimensionVector> result;

    for (const auto kernel : kernels)
    {
        DimensionVector kernelDimensions = kernel->getGlobalSize();

        if (globalThreadModifiers.find(kernel->getId()) != globalThreadModifiers.end())
        {
            auto kernelGlobalThreadModifiers = globalThreadModifiers.find(kernel->getId())->second;
            auto kernelGlobalThreadModifierNames = globalThreadModifierNames.find(kernel->getId())->second;

            for (size_t i = 0; i < 3; i++)
            {
                std::vector<size_t> parameterValues;

                for (const auto& parameterName : kernelGlobalThreadModifierNames[i])
                {
                    for (const auto& parameterPair : parameterPairs)
                    {
                        if (parameterName == parameterPair.getName())
                        {
                            parameterValues.push_back(parameterPair.getValue());
                            break;
                        }
                    }
                }

                if (kernelGlobalThreadModifiers[i] != nullptr)
                {
                    kernelDimensions.setSize(static_cast<ModifierDimension>(i),
                        kernelGlobalThreadModifiers[i](kernel->getGlobalSize().getSize(static_cast<ModifierDimension>(i)), parameterValues));
                }
            }
        }

        result.insert(std::make_pair(kernel->getId(), kernelDimensions));
    }

    return result;
}

std::map<KernelId, DimensionVector> KernelComposition::getModifiedLocalSizes(const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName()
                + " is not associated with kernel composition with id " + std::to_string(id));
        }
    }

    std::map<KernelId, DimensionVector> result;

    for (const auto kernel : kernels)
    {
        DimensionVector kernelDimensions = kernel->getLocalSize();

        if (localThreadModifiers.find(kernel->getId()) != localThreadModifiers.end())
        {
            auto kernelLocalThreadModifiers = localThreadModifiers.find(kernel->getId())->second;
            auto kernelLocalThreadModifierNames = localThreadModifierNames.find(kernel->getId())->second;

            for (size_t i = 0; i < 3; i++)
            {
                std::vector<size_t> parameterValues;

                for (const auto& parameterName : kernelLocalThreadModifierNames[i])
                {
                    for (const auto& parameterPair : parameterPairs)
                    {
                        if (parameterName == parameterPair.getName())
                        {
                            parameterValues.push_back(parameterPair.getValue());
                            break;
                        }
                    }
                }

                if (kernelLocalThreadModifiers[i] != nullptr)
                {
                    kernelDimensions.setSize(static_cast<ModifierDimension>(i),
                        kernelLocalThreadModifiers[i](kernel->getLocalSize().getSize(static_cast<ModifierDimension>(i)), parameterValues));
                }
            }
        }

        result.insert(std::make_pair(kernel->getId(), kernelDimensions));
    }

    return result;
}

std::map<KernelId, std::vector<LocalMemoryModifier>> KernelComposition::getLocalMemoryModifiers(
    const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName()
                + " is not associated with kernel composition with id " + std::to_string(id));
        }
    }

    std::map<KernelId, std::vector<LocalMemoryModifier>> result;

    for (const auto kernel : kernels)
    {
        std::vector<LocalMemoryModifier> kernelModifiers;

        if (localMemoryModifiers.find(kernel->getId()) != localMemoryModifiers.end())
        {
            auto kernelLocalMemoryModifiers = localMemoryModifiers.find(kernel->getId())->second;
            auto kernelLocalMemoryModifierNames = localMemoryModifierNames.find(kernel->getId())->second;

            for (const auto& modifier : kernelLocalMemoryModifiers)
            {
                std::vector<size_t> parameterValues;
                std::vector<std::string> parameterNames = kernelLocalMemoryModifierNames.find(modifier.first)->second;

                for (const auto& name : parameterNames)
                {
                    for (const auto& pair : parameterPairs)
                    {
                        if (name == pair.getName())
                        {
                            parameterValues.push_back(pair.getValue());
                            break;
                        }
                    }
                }

                kernelModifiers.emplace_back(id, modifier.first, parameterValues, modifier.second);
            }
        }

        result.insert(std::make_pair(kernel->getId(), kernelModifiers));
    }

    return result;
}

void KernelComposition::setKernelProfiling(const KernelId kernelId, const bool flag)
{
    bool kernelFound = false;

    for (const auto* kernel : kernels)
    {
        if (kernel->getId() == kernelId)
        {
            kernelFound = true;
            break;
        }
    }

    if (!kernelFound)
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " is not included in composition with the following id: " + std::to_string(id));
    }

    if (flag)
    {
        profiledKernels.insert(kernelId);
    }
    else
    {
        profiledKernels.erase(kernelId);
    }
}

KernelId KernelComposition::getId() const
{
    return id;
}

const std::string& KernelComposition::getName() const
{
    return name;
}

const std::vector<const Kernel*>& KernelComposition::getKernels() const
{
    return kernels;
}

const std::vector<KernelParameter>& KernelComposition::getParameters() const
{
    return parameters;
}

const std::vector<KernelConstraint>& KernelComposition::getConstraints() const
{
    return constraints;
}

const std::vector<KernelParameterPack>& KernelComposition::getParameterPacks() const
{
    return parameterPacks;
}

std::vector<KernelParameter> KernelComposition::getParametersOutsidePacks() const
{
    std::vector<KernelParameter> result;

    for (const auto& parameter : parameters)
    {
        bool isOutsidePack = true;

        for (const auto& pack : parameterPacks)
        {
            isOutsidePack &= !pack.containsParameter(parameter.getName());
        }

        if (isOutsidePack)
        {
            result.push_back(parameter);
        }
    }

    return result;
}

std::vector<KernelParameter> KernelComposition::getParametersForPack(const std::string& pack) const
{
    KernelParameterPack searchPack(pack, std::vector<std::string>{});
    return getParametersForPack(searchPack);
}

std::vector<KernelParameter> KernelComposition::getParametersForPack(const KernelParameterPack& pack) const
{
    auto targetPack = std::find(std::begin(parameterPacks), std::end(parameterPacks), pack);

    if (targetPack == std::end(parameterPacks))
    {
        throw std::runtime_error(std::string("The following parameter pack does not exist: ") + pack.getName());
    }

    std::vector<KernelParameter> result;

    for (const auto& parameter : parameters)
    {
        if (targetPack->containsParameter(parameter.getName()))
        {
            result.push_back(parameter);
        }
    }

    return result;
}

const std::vector<ArgumentId>& KernelComposition::getSharedArgumentIds() const
{
    return sharedArgumentIds;
}

const std::vector<ArgumentId>& KernelComposition::getKernelArgumentIds(const KernelId id) const
{
    auto pointer = kernelArgumentIds.find(id);
    if (pointer != kernelArgumentIds.end())
    {
        return pointer->second;
    }
    return sharedArgumentIds;
}

const std::set<KernelId>& KernelComposition::getProfiledKernels() const
{
    return profiledKernels;
}

bool KernelComposition::hasParameter(const std::string& parameterName) const
{
    for (const auto& currentParameter : parameters)
    {
        if (currentParameter.getName() == parameterName)
        {
            return true;
        }
    }
    return false;
}

void KernelComposition::validateModifierParameters(const std::vector<std::string>& parameterNames) const
{
    for (const auto& parameterName : parameterNames)
    {
        bool parameterFound = false;

        for (const auto& parameter : parameters)
        {
            if (parameter.getName() == parameterName)
            {
                if (parameter.hasValuesDouble())
                {
                    throw std::runtime_error("Parameters with floating-point values cannot act as thread modifiers");
                }

                parameterFound = true;
                break;
            }
        }

        if (!parameterFound)
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterName + " does not exist");
        }
    }
}

} // namespace ktt
