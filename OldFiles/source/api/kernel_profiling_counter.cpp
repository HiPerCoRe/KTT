#include <api/kernel_profiling_counter.h>

namespace ktt
{

KernelProfilingCounter::KernelProfilingCounter(const std::string& name, const ProfilingCounterValue& value, const ProfilingCounterType type) :
    name(name),
    value(value),
    type(type)
{}

const std::string& KernelProfilingCounter::getName() const
{
    return name;
}

const ProfilingCounterValue& KernelProfilingCounter::getValue() const
{
    return value;
}

ProfilingCounterType KernelProfilingCounter::getType() const
{
    return type;
}

} // namespace ktt
