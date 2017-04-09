#include "kernel_parameter.h"

namespace ktt
{

KernelParameter::KernelParameter(const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
    const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension) :
    name(name),
    values(values),
    threadModifierType(threadModifierType),
    threadModifierAction(threadModifierAction),
    modifierDimension(modifierDimension)
{}

std::string KernelParameter::getName() const
{
    return name;
}

std::vector<size_t> KernelParameter::getValues() const
{
    return values;
}

ThreadModifierType KernelParameter::getThreadModifierType() const
{
    return threadModifierType;
}

ThreadModifierAction KernelParameter::getThreadModifierAction() const
{
    return threadModifierAction;
}

Dimension KernelParameter::getModifierDimension() const
{
    return modifierDimension;
}

} // namespace ktt
