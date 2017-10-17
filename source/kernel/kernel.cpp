#include <stdexcept>

#include "kernel.h"
#include "utility/ktt_utility.h"

namespace ktt
{

Kernel::Kernel(const size_t id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
    const DimensionVector& localSize) :
    id(id),
    source(source),
    name(name),
    globalSize(globalSize),
    localSize(localSize),
    tuningManipulatorFlag(false)
{}

void Kernel::addParameter(const KernelParameter& parameter)
{
    if (elementExists(parameter, parameters))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: ") + parameter.getName());
    }
    parameters.push_back(parameter);
}

void Kernel::addConstraint(const KernelConstraint& constraint)
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

void Kernel::setArguments(const std::vector<size_t>& argumentIndices)
{
    this->argumentIndices = argumentIndices;
}

void Kernel::setTuningManipulatorFlag(const bool tuningManipulatorFlag)
{
    this->tuningManipulatorFlag = tuningManipulatorFlag;
}

size_t Kernel::getId() const
{
    return id;
}

std::string Kernel::getSource() const
{
    return source;
}

std::string Kernel::getName() const
{
    return name;
}

DimensionVector Kernel::getGlobalSize() const
{
    return globalSize;
}

DimensionVector Kernel::getLocalSize() const
{
    return localSize;
}

std::vector<KernelParameter> Kernel::getParameters() const
{
    return parameters;
}

std::vector<KernelConstraint> Kernel::getConstraints() const
{
    return constraints;
}

size_t Kernel::getArgumentCount() const
{
    return argumentIndices.size();
}

std::vector<size_t> Kernel::getArgumentIndices() const
{
    return argumentIndices;
}

bool Kernel::hasParameter(const std::string& parameterName) const
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

bool Kernel::hasTuningManipulator() const
{
    return tuningManipulatorFlag;
}

} // namespace ktt
