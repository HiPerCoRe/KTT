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
    if (hasParameter(parameter.getName()))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }

    parameters.push_back(parameter);
}

void KernelComposition::addConstraint(const KernelConstraint& constraint)
{
    auto parameterNames = constraint.getParameterNames();

    for (const auto& parameterName : parameterNames)
    {
        if (!hasParameter(parameterName))
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
        }
    }

    constraints.push_back(constraint);
}

void KernelComposition::setSharedArguments(const std::vector<size_t>& argumentIds)
{
    this->sharedArgumentIds = argumentIds;
}

void KernelComposition::addKernelParameter(const size_t kernelId, const KernelParameter& parameter)
{
    if (!hasParameter(parameter.getName()))
    {
        parameters.push_back(parameter);
    }

    KernelParameter* targetParameter;
    for (auto& existingParameter : parameters)
    {
        if (existingParameter == parameter)
        {
            targetParameter = &existingParameter;
            break;
        }
    }

    if (parameter.getThreadModifierAction() != targetParameter->getThreadModifierAction()
        || parameter.getThreadModifierType() != targetParameter->getThreadModifierType()
        || parameter.getModifierDimension() != targetParameter->getModifierDimension()
        || parameter.getValues().size() != targetParameter->getValues().size())
    {
        throw std::runtime_error("Composition parameters with same name must have matching thread modifier properties");
    }

    for (size_t i = 0; i < parameter.getValues().size(); i++)
    {
        if (parameter.getValues().at(i) != targetParameter->getValues().at(i))
        {
            throw std::runtime_error("Composition parameters with same name must have matching thread modifier properties");
        }
    }

    for (const auto currentKernelId : targetParameter->getCompositionKernels())
    {
        if (currentKernelId == kernelId)
        {
            throw std::runtime_error(std::string("Composition parameter with name ") + targetParameter->getName()
                + " already affects kernel with id: " + std::to_string(kernelId));
        }
    }

    targetParameter->addCompositionKernel(kernelId);
}

void KernelComposition::setKernelArguments(const size_t id, const std::vector<size_t>& argumentIds)
{
    if (kernelArgumentIds.find(id) != kernelArgumentIds.end())
    {
        kernelArgumentIds.erase(id);
    }
    kernelArgumentIds.insert(std::make_pair(id, argumentIds));
}

size_t KernelComposition::getId() const
{
    return id;
}

std::vector<Kernel*> KernelComposition::getKernels() const
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

std::vector<size_t> KernelComposition::getSharedArgumentIds() const
{
    return sharedArgumentIds;
}

std::vector<size_t> KernelComposition::getKernelArgumentIds(const size_t kernelId) const
{
    auto pointer = kernelArgumentIds.find(kernelId);
    if (pointer != kernelArgumentIds.end())
    {
        return pointer->second;
    }
    return sharedArgumentIds;
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
