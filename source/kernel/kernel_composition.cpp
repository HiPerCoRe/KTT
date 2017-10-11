#include <stdexcept>

#include "kernel_composition.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelComposition::KernelComposition(const size_t id, std::vector<const Kernel*> kernels) :
    id(id),
    kernels(kernels),
    compositeArguments(false)
{}

void KernelComposition::addParameter(const KernelParameter& parameter)
{
    if (elementExists(parameter, parameters))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }
    parameters.push_back(parameter);
}

void KernelComposition::addConstraint(const KernelConstraint& constraint)
{
    auto parameterNames = constraint.getParameterNames();
    std::vector<size_t> affectedKernels;

    for (const auto& parameterName : parameterNames)
    {
        bool parameterFound = false;
        parameterFound |= hasParameter(parameterName);

        for (const auto& kernel : kernels)
        {
            if (kernel->hasParameter(parameterName))
            {
                affectedKernels.push_back(kernel->getId());
                parameterFound = true;
            }
        }

        if (!parameterFound)
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
        }
    }
    constraints.push_back(constraint);
    kernelsWithConstraint.push_back(affectedKernels);
}

void KernelComposition::setArguments(const std::vector<size_t>& argumentIds)
{
    this->argumentIds = argumentIds;
    compositeArguments = true;
}

size_t KernelComposition::getId() const
{
    return id;
}

std::vector<const Kernel*> KernelComposition::getKernels() const
{
    return kernels;
}

std::vector<KernelParameter> KernelComposition::getParameters() const
{
    return parameters;
}

std::vector<KernelConstraint> KernelComposition::getConstraints() const
{
    return constraints;
}

std::vector<KernelConstraint> KernelComposition::getConstraintsForKernel(const size_t id) const
{
    std::vector<KernelConstraint> result;

    for (size_t i = 0; i < constraints.size(); i++)
    {
        for (const auto& kernelId : kernelsWithConstraint.at(i))
        {
            if (kernelId == id)
            {
                result.push_back(constraints.at(i));
                continue;
            }
        }
    }

    return result;
}

std::vector<size_t> KernelComposition::getArgumentIds() const
{
    return argumentIds;
}

bool KernelComposition::hasCompositeArguments() const
{
    return compositeArguments;
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

} // namespace ktt
