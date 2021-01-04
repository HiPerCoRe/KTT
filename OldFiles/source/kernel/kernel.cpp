#include <algorithm>
#include <stdexcept>
#include <kernel/kernel.h>
#include <utility/ktt_utility.h>

namespace ktt
{

Kernel::Kernel(const KernelId id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
    const DimensionVector& localSize) :
    id(id),
    source(source),
    name(name),
    globalSize(globalSize),
    localSize(localSize),
    globalThreadModifiers{nullptr, nullptr, nullptr},
    localThreadModifiers{nullptr, nullptr, nullptr},
    tuningManipulatorFlag(false)
{}

void Kernel::addParameter(const KernelParameter& parameter)
{
    if (containsElement(parameters, parameter))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }
    parameters.push_back(parameter);
}

void Kernel::addConstraint(const KernelConstraint& constraint)
{
    auto parameterNames = constraint.getParameterNames();

    for (const auto& parameterName : parameterNames)
    {
        if (!hasParameter(parameterName))
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
        }
    }
    constraints.push_back(constraint);
}

void Kernel::addParameterPack(const KernelParameterPack& pack)
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

void Kernel::setThreadModifier(const ModifierType modifierType, const ModifierDimension modifierDimension,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    validateModifierParameters(parameterNames);
    
    switch (modifierType)
    {
    case ModifierType::Global:
        globalThreadModifiers[static_cast<size_t>(modifierDimension)] = modifierFunction;
        globalThreadModifierNames[static_cast<size_t>(modifierDimension)] = parameterNames;
        break;
    case ModifierType::Local:
        localThreadModifiers[static_cast<size_t>(modifierDimension)] = modifierFunction;
        localThreadModifierNames[static_cast<size_t>(modifierDimension)] = parameterNames;
        break;
    default:
        throw std::runtime_error("Unknown modifier type");
    }
}

void Kernel::setLocalMemoryModifier(const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    validateModifierParameters(parameterNames);

    if (localMemoryModifiers.find(argumentId) != localMemoryModifiers.end())
    {
        localMemoryModifiers.erase(argumentId);
        localMemoryModifierNames.erase(argumentId);
    }
    localMemoryModifiers.insert(std::make_pair(argumentId, modifierFunction));
    localMemoryModifierNames.insert(std::make_pair(argumentId, parameterNames));
}

void Kernel::setArguments(const std::vector<ArgumentId>& argumentIds)
{
    this->argumentIds = argumentIds;
}

void Kernel::setTuningManipulatorFlag(const bool flag)
{
    this->tuningManipulatorFlag = flag;
}

uint64_t Kernel::getConfigurationsCount() const
{
    uint64_t result = 1;

    for (const auto& parameter : parameters)
    {
        result *= parameter.getValues().size();
    }

    return result;
}

std::vector<ParameterPair> Kernel::getConfigurationForIndex(const uint64_t index) const
{
    std::vector<ParameterPair> result;
    uint64_t currentIndex = index;

    for (const auto& parameter : parameters)
    {
        const size_t valuesCount = parameter.getValues().size();
        const size_t parameterIndex = currentIndex % valuesCount;

        if (parameter.hasValuesDouble())
        {
            result.emplace_back(parameter.getName(), parameter.getValuesDouble()[parameterIndex]);
        }
        else
        {
            result.emplace_back(parameter.getName(), parameter.getValues()[parameterIndex]);
        }

        currentIndex /= valuesCount;
    }

    return result;
}

uint64_t Kernel::getIndexForConfiguration(const std::vector<ParameterPair>& configuration) const
{
    uint64_t result = 0;
    uint64_t multiplier = 1;

    for (const auto& parameter : parameters)
    {
        const ParameterPair* currentPair = nullptr;

        for (const auto& pair : configuration)
        {
            if (parameter.getName() == pair.getName())
            {
                currentPair = &pair;
            }
        }

        if (currentPair == nullptr)
        {
            throw std::runtime_error(std::string("Parameter ") + parameter.getName() + " was not found in provided kernel configuration");
        }

        const size_t valuesCount = parameter.getValues().size();

        for (size_t i = 0; i < valuesCount; ++i)
        {
            if ((parameter.hasValuesDouble() && floatEquals(parameter.getValuesDouble()[i], currentPair->getValueDouble()))
                || (!parameter.hasValuesDouble() && parameter.getValues()[i] == currentPair->getValue()))
            {
                result += multiplier * i;
                break;
            }
        }

        multiplier *= valuesCount;
    }

    return result;
}

KernelId Kernel::getId() const
{
    return id;
}

const std::string& Kernel::getSource() const
{
    return source;
}

const std::string& Kernel::getName() const
{
    return name;
}

const DimensionVector& Kernel::getGlobalSize() const
{
    return globalSize;
}

DimensionVector Kernel::getModifiedGlobalSize(const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName() + " is not associated with kernel with id "
                + std::to_string(id));
        }
    }

    DimensionVector result = getGlobalSize();

    for (size_t i = 0; i < 3; i++)
    {
        std::vector<size_t> parameterValues;

        for (const auto& parameterName : globalThreadModifierNames[i])
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

        if (globalThreadModifiers[i] != nullptr)
        {
            result.setSize(static_cast<ModifierDimension>(i), globalThreadModifiers[i](globalSize.getSize(static_cast<ModifierDimension>(i)),
                parameterValues));
        }
    }

    return result;
}

const DimensionVector& Kernel::getLocalSize() const
{
    return localSize;
}

DimensionVector Kernel::getModifiedLocalSize(const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName() + " is not associated with kernel with id "
                + std::to_string(id));
        }
    }

    DimensionVector result = getLocalSize();

    for (size_t i = 0; i < 3; i++)
    {
        std::vector<size_t> parameterValues;

        for (const auto& parameterName : localThreadModifierNames[i])
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

        if (localThreadModifiers[i] != nullptr)
        {
            result.setSize(static_cast<ModifierDimension>(i), localThreadModifiers[i](localSize.getSize(static_cast<ModifierDimension>(i)),
                parameterValues));
        }
    }

    return result;
}

const std::vector<KernelParameter>& Kernel::getParameters() const
{
    return parameters;
}

const std::vector<KernelConstraint>& Kernel::getConstraints() const
{
    return constraints;
}

const std::vector<KernelParameterPack>& Kernel::getParameterPacks() const
{
    return parameterPacks;
}

std::vector<KernelParameter> Kernel::getParametersOutsidePacks() const
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

std::vector<KernelParameter> Kernel::getParametersForPack(const std::string& pack) const
{
    KernelParameterPack searchPack(pack, std::vector<std::string>{});
    return getParametersForPack(searchPack);
}

std::vector<KernelParameter> Kernel::getParametersForPack(const KernelParameterPack& pack) const
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

size_t Kernel::getArgumentCount() const
{
    return argumentIds.size();
}

const std::vector<ArgumentId>& Kernel::getArgumentIds() const
{
    return argumentIds;
}

std::vector<LocalMemoryModifier> Kernel::getLocalMemoryModifiers(const std::vector<ParameterPair>& parameterPairs) const
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (!hasParameter(parameterPair.getName()))
        {
            throw std::runtime_error(std::string("Parameter with name ") + parameterPair.getName() + " is not associated with kernel with id "
                + std::to_string(id));
        }
    }

    std::vector<LocalMemoryModifier> result;
    for (const auto& modifier : localMemoryModifiers)
    {
        std::vector<size_t> parameterValues;
        std::vector<std::string> parameterNames = localMemoryModifierNames.find(modifier.first)->second;

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

        result.emplace_back(id, modifier.first, parameterValues, modifier.second);
    }

    return result;
}

bool Kernel::hasParameter(const std::string& parameterName) const
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

bool Kernel::hasTuningManipulator() const
{
    return tuningManipulatorFlag;
}

void Kernel::validateModifierParameters(const std::vector<std::string>& parameterNames) const
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
