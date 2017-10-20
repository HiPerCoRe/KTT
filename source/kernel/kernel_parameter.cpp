#include "kernel_parameter.h"

namespace ktt
{

KernelParameter::KernelParameter(const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& modifierType,
    const ThreadModifierAction& modifierAction, const Dimension& modifierDimension) :
    name(name),
    values(values),
    threadModifierType(modifierType),
    threadModifierAction(modifierAction),
    modifierDimension(modifierDimension)
{}

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

ThreadModifierType KernelParameter::getModifierType() const
{
    return threadModifierType;
}

ThreadModifierAction KernelParameter::getModifierAction() const
{
    return threadModifierAction;
}

Dimension KernelParameter::getModifierDimension() const
{
    return modifierDimension;
}

std::vector<KernelId> KernelParameter::getCompositionKernels() const
{
    return compositionKernels;
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
