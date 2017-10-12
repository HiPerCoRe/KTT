#include <stdexcept>

#include "kernel_composition.h"

namespace ktt
{

KernelComposition::KernelComposition(const size_t id, const std::vector<Kernel*>& kernels) :
    id(id),
    kernels(kernels)
{}

void KernelComposition::addParameter(const KernelParameter& parameter)
{
    for (auto& kernel : kernels)
    {
        if (kernel->hasParameter(parameter.getName()))
        {
            throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
        }
        kernel->addParameter(parameter);
    }
}

void KernelComposition::addConstraint(const KernelConstraint& constraint)
{
    auto parameterNames = constraint.getParameterNames();

    for (auto& kernel : kernels)
    {
        for (const auto& parameterName : parameterNames)
        {
            if (!kernel->hasParameter(parameterName))
            {
                throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
            }
        }
        kernel->addConstraint(constraint);
    }
}

void KernelComposition::setArguments(const std::vector<size_t>& argumentIds)
{
    for (auto& kernel : kernels)
    {
        kernel->setArguments(argumentIds);
    }
}

size_t KernelComposition::getId() const
{
    return id;
}

std::vector<Kernel*> KernelComposition::getKernels() const
{
    return kernels;
}

} // namespace ktt
