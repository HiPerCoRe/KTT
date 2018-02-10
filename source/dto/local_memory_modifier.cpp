#include "local_memory_modifier.h"

namespace ktt
{

LocalMemoryModifier::LocalMemoryModifier() :
    LocalMemoryModifier(0, 0, ModifierAction::Add, 0)
{}

LocalMemoryModifier::LocalMemoryModifier(const KernelId kernel, const ArgumentId argument, const ModifierAction action, const size_t value) :
    kernel(kernel),
    argument(argument),
    action(action),
    value(value)
{}

void LocalMemoryModifier::setAction(const ModifierAction action)
{
    this->action = action;
}

void LocalMemoryModifier::setValue(const size_t value)
{
    this->value = value;
}

KernelId LocalMemoryModifier::getKernel() const
{
    return kernel;
}

ArgumentId LocalMemoryModifier::getArgument() const
{
    return argument;
}

ModifierAction LocalMemoryModifier::getAction() const
{
    return action;
}

size_t LocalMemoryModifier::getValue() const
{
    return value;
}

} // namespace ktt
