#include "kernel_parameter.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelParameter::KernelParameter(const std::string& name, const std::vector<size_t>& values, const ModifierType modifierType,
    const ModifierAction modifierAction, const ModifierDimension modifierDimension) :
    name(name),
    values(values),
    modifierType(modifierType),
    modifierAction(modifierAction),
    modifierDimension(modifierDimension),
    isDouble(false),
    localMemoryModifierFlag(false)
{}

KernelParameter::KernelParameter(const std::string& name, const std::vector<double>& values) :
    name(name),
    valuesDouble(values),
    modifierType(ModifierType::None),
    modifierAction(ModifierAction::Add),
    modifierDimension(ModifierDimension::X),
    isDouble(true),
    localMemoryModifierFlag(false)
{}

void KernelParameter::setLocalMemoryArgumentModifier(const ArgumentId id, const ModifierAction modifierAction)
{
    for (auto& argumentPair : localMemoryArguments)
    {
        if (id == argumentPair.first)
        {
            argumentPair.second = modifierAction;
            return;
        }
    }

    localMemoryArguments.push_back(std::make_pair(id, modifierAction));
    localMemoryModifierFlag = true;
}

void KernelParameter::setLocalMemoryArgumentModifier(const KernelId compositionKernelId, ArgumentId id, const ModifierAction modifierAction)
{
    if (!elementExists(compositionKernelId, localMemoryModifierKernels))
    {
        localMemoryModifierKernels.push_back(compositionKernelId);
    }

    setLocalMemoryArgumentModifier(id, modifierAction);
}

void KernelParameter::addCompositionKernel(const KernelId id)
{
    compositionKernels.push_back(static_cast<size_t>(id));
}

std::string KernelParameter::getName() const
{
    return name;
}

std::vector<size_t> KernelParameter::getValues() const
{
    return values;
}

std::vector<double> KernelParameter::getValuesDouble() const
{
    return valuesDouble;
}

ModifierType KernelParameter::getModifierType() const
{
    return modifierType;
}

ModifierAction KernelParameter::getModifierAction() const
{
    return modifierAction;
}

ModifierDimension KernelParameter::getModifierDimension() const
{
    return modifierDimension;
}

std::vector<KernelId> KernelParameter::getCompositionKernels() const
{
    return compositionKernels;
}

bool KernelParameter::hasValuesDouble() const
{
    return isDouble;
}

bool KernelParameter::isLocalMemoryModifier() const
{
    return localMemoryModifierFlag;
}

std::vector<std::pair<ArgumentId, ModifierAction>> KernelParameter::getLocalMemoryArguments() const
{
    return localMemoryArguments;
}

std::vector<KernelId> KernelParameter::getLocalMemoryModifierKernels() const
{
    return localMemoryModifierKernels;
}

bool KernelParameter::operator==(const KernelParameter& other) const
{
    return name == other.name;
}

bool KernelParameter::operator!=(const KernelParameter& other) const
{
    return !(*this == other);
}

} // namespace ktt
