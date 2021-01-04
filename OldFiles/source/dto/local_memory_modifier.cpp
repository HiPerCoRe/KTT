#include <stdexcept>
#include <dto/local_memory_modifier.h>

namespace ktt
{

LocalMemoryModifier::LocalMemoryModifier() :
    LocalMemoryModifier(0, 0, std::vector<size_t>{}, nullptr)
{}

LocalMemoryModifier::LocalMemoryModifier(const KernelId kernel, const ArgumentId argument, const std::vector<size_t>& parameterValues,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction) :
    kernel(kernel),
    argument(argument),
    parameterValues(parameterValues),
    modifierFunction(modifierFunction)
{}

KernelId LocalMemoryModifier::getKernel() const
{
    return kernel;
}

ArgumentId LocalMemoryModifier::getArgument() const
{
    return argument;
}

const std::vector<size_t>& LocalMemoryModifier::getParameterValues() const
{
    return parameterValues;
}

std::function<size_t(const size_t, const std::vector<size_t>&)> LocalMemoryModifier::getModifierFunction() const
{
    return modifierFunction;
}

size_t LocalMemoryModifier::getModifiedSize(const size_t defaultSize) const
{
    if (modifierFunction == nullptr)
    {
        return defaultSize;
    }

    return modifierFunction(defaultSize, parameterValues);
}

} // namespace ktt
