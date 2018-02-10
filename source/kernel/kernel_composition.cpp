#include <stdexcept>
#include "kernel_composition.h"

namespace ktt
{

KernelComposition::KernelComposition(const KernelId id, const std::string& name, const std::vector<const Kernel*>& kernels) :
    id(id),
    name(name),
    kernels(kernels)
{}

void KernelComposition::addParameter(const KernelParameter& parameter)
{
    if (hasParameter(parameter.getName()))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }

    KernelParameter parameterCopy = parameter;
    if (parameter.getModifierType() != ModifierType::None)
    {
        for (const auto kernel : kernels)
        {
            parameterCopy.addCompositionKernel(kernel->getId());
        }
    }

    parameters.push_back(parameterCopy);
}

void KernelComposition::addLocalMemoryModifier(const std::string& parameterName, const ArgumentId argumentId, const ModifierAction& modifierAction)
{
    if (!hasParameter(parameterName))
    {
        throw std::runtime_error(std::string("Parameter with name does not exist: ") + parameterName);
    }

    for (auto& parameter : parameters)
    {
        if (parameter.getName() == parameterName)
        {
            parameter.setLocalMemoryArgumentModifier(argumentId, modifierAction);
            return;
        }
    }
}

void KernelComposition::addConstraint(const KernelConstraint& constraint)
{
    std::vector<std::string> parameterNames = constraint.getParameterNames();

    for (const auto& parameterName : parameterNames)
    {
        if (!hasParameter(parameterName))
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: ") + parameterName);
        }
    }

    constraints.push_back(constraint);
}

void KernelComposition::setSharedArguments(const std::vector<ArgumentId>& argumentIds)
{
    this->sharedArgumentIds = argumentIds;
}

void KernelComposition::addKernelParameter(const KernelId id, const KernelParameter& parameter)
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

    if (parameter.getModifierAction() != targetParameter->getModifierAction()
        || parameter.getModifierType() != targetParameter->getModifierType()
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
        if (currentKernelId == id)
        {
            throw std::runtime_error(std::string("Composition parameter with name ") + targetParameter->getName()
                + " already affects kernel with id: " + std::to_string(id));
        }
    }

    targetParameter->addCompositionKernel(id);
}

void KernelComposition::addKernelLocalMemoryModifier(const KernelId id, const std::string& parameterName, const ArgumentId argumentId,
    const ModifierAction& modifierAction)
{
    if (!hasParameter(parameterName))
    {
        throw std::runtime_error(std::string("Parameter with name does not exist: ") + parameterName);
    }

    for (auto& parameter : parameters)
    {
        if (parameter.getName() == parameterName)
        {
            parameter.setLocalMemoryArgumentModifier(id, argumentId, modifierAction);
            return;
        }
    }
}

void KernelComposition::setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    if (kernelArgumentIds.find(id) != kernelArgumentIds.end())
    {
        kernelArgumentIds.erase(id);
    }
    kernelArgumentIds.insert(std::make_pair(id, argumentIds));
}

Kernel KernelComposition::transformToKernel() const
{
    Kernel kernel(id, "", name, DimensionVector(), DimensionVector());
    kernel.setTuningManipulatorFlag(true);

    for (const auto& constraint : constraints)
    {
        kernel.addConstraint(constraint);
    }

    for (const auto& parameter : parameters)
    {
        kernel.addParameter(parameter);
    }

    std::vector<size_t> argumentIds;
    for (const auto id : sharedArgumentIds)
    {
        argumentIds.push_back(id);
    }

    for (const auto& kernel : kernels)
    {
        std::vector<ArgumentId> kernelArgumentIds = getKernelArgumentIds(kernel->getId());
        for (const auto id : kernelArgumentIds)
        {
            argumentIds.push_back(id);
        }
    }
    kernel.setArguments(argumentIds);

    return kernel;
}

KernelId KernelComposition::getId() const
{
    return id;
}

std::string KernelComposition::getName() const
{
    return name;
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

std::vector<ArgumentId> KernelComposition::getSharedArgumentIds() const
{
    return sharedArgumentIds;
}

std::vector<ArgumentId> KernelComposition::getKernelArgumentIds(const KernelId id) const
{
    auto pointer = kernelArgumentIds.find(id);
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
