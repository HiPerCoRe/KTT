#include <kernel/kernel_constraint.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<std::string>& parameterNames,
    const std::function<bool(const std::vector<size_t>&)>& constraintFunction) :
    parameterNames(parameterNames),
    constraintFunction(constraintFunction)
{}

const std::vector<std::string>& KernelConstraint::getParameterNames() const
{
    return parameterNames;
}

std::function<bool(const std::vector<size_t>&)> KernelConstraint::getConstraintFunction() const
{
    return constraintFunction;
}

} // namespace ktt
