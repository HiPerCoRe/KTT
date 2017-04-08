#include "kernel_constraint.h"

namespace ktt
{

KernelConstraint::KernelConstraint(const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames):
    constraintFunction(constraintFunction),
    parameterNames(parameterNames)
{}

std::function<bool(std::vector<size_t>)> KernelConstraint::getConstraintFunction() const
{
    return constraintFunction;
}

std::vector<std::string> KernelConstraint::getParameterNames() const
{
    return parameterNames;
}

} // namespace ktt
