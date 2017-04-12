#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager() :
    kernelCount(0)
{}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    kernels.emplace_back(Kernel(kernelCount, source, kernelName, globalSize, localSize));
    return kernelCount++;
}

size_t KernelManager::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filePath);
    return addKernel(source, kernelName, globalSize, localSize);
}

std::string KernelManager::getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const
{
    std::string source = kernels.at(id).getSource();

    for (const auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterValue); // clean way to convert number to string
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
}

KernelConfiguration KernelManager::getKernelConfiguration(const size_t id, const std::vector<ParameterValue>& parameterValues) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }

    DimensionVector global = kernels.at(id).getGlobalSize();
    DimensionVector local = kernels.at(id).getLocalSize();
    
    for (const auto& value : parameterValues)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernels.at(id).getParameters())
        {
            if (parameter.getName() == std::get<0>(value))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <" + std::get<0>(value) + "> is not associated with kernel with id: " +
                std::to_string(id)));
        }

        global = modifyDimensionVector(global, DimensionVectorType::Global, *targetParameter, std::get<1>(value));
        local = modifyDimensionVector(local, DimensionVectorType::Local, *targetParameter, std::get<1>(value));
    }

    return KernelConfiguration(global, local, parameterValues);
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurations(const size_t id) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }

    std::vector<KernelConfiguration> configurations;

    if (kernels.at(id).getParameters().size() == 0)
    {
        configurations.emplace_back(KernelConfiguration(kernels.at(id).getGlobalSize(), kernels.at(id).getLocalSize(),
            std::vector<ParameterValue>{}));
    }
    else
    {
        computeConfigurations(0, kernels.at(id).getParameters(), kernels.at(id).getConstraints(), std::vector<ParameterValue>(0),
            kernels.at(id).getGlobalSize(), kernels.at(id).getLocalSize(), configurations);
    }
    return configurations;
}

void KernelManager::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    kernels.at(id).addParameter(KernelParameter(name, values, threadModifierType, threadModifierAction, modifierDimension));
}

void KernelManager::addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    kernels.at(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
}

void KernelManager::setArguments(const size_t id, const std::vector<size_t>& argumentIndices)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    kernels.at(id).setArguments(argumentIndices);
}

void KernelManager::setSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    kernels.at(id).setSearchMethod(searchMethod, searchArguments);
}

void KernelManager::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    if (kernelId >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(kernelId)));
    }
    if (referenceKernelId >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid reference kernel id: " + std::to_string(referenceKernelId)));
    }
    kernels.at(kernelId).setReferenceKernel(referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
}

void KernelManager::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    if (kernelId >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(kernelId)));
    }
    kernels.at(kernelId).setReferenceClass(std::move(referenceClass), resultArgumentIds);
}

size_t KernelManager::getKernelCount() const
{
    return kernelCount;
}

const Kernel* KernelManager::getKernel(const size_t id) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    return &kernels.at(id);
}

std::string KernelManager::loadFileToString(const std::string& filePath) const
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: " + filePath));
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

void KernelManager::computeConfigurations(const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
    const std::vector<KernelConstraint>& constraints, const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize,
    const DimensionVector& localSize, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterValues);
        if (configurationIsValid(configuration, constraints))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterValues = parameterValues;
        newParameterValues.push_back(ParameterValue(parameter.getName(), value));

        auto newGlobalSize = modifyDimensionVector(globalSize, DimensionVectorType::Global, parameter, value);
        auto newLocalSize = modifyDimensionVector(localSize, DimensionVectorType::Local, parameter, value);

        computeConfigurations(currentParameterIndex + 1, parameters, constraints, newParameterValues, newGlobalSize, newLocalSize, finalResult);
    }
}

DimensionVector KernelManager::modifyDimensionVector(const DimensionVector& vector, const DimensionVectorType& dimensionVectorType,
    const KernelParameter& parameter, const size_t parameterValue) const
{
    if (parameter.getThreadModifierType() == ThreadModifierType::None
        || dimensionVectorType == DimensionVectorType::Global && parameter.getThreadModifierType() == ThreadModifierType::Local
        || dimensionVectorType == DimensionVectorType::Local && parameter.getThreadModifierType() == ThreadModifierType::Global)
    {
        return vector;
    }

    ThreadModifierAction action = parameter.getThreadModifierAction();
    if (action == ThreadModifierAction::Add)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) + parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) + parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) + parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else if (action == ThreadModifierAction::Subtract)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) - parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) - parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) - parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else if (action == ThreadModifierAction::Multiply)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) * parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) * parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) * parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else // divide
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) / parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) / parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) / parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
}

bool KernelManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const
{
    for (const auto& constraint : constraints)
    {
        std::vector<std::string> constraintNames = constraint.getParameterNames();
        auto constraintValues = std::vector<size_t>(constraintNames.size());

        for (size_t i = 0; i < constraintNames.size(); i++)
        {
            for (const auto& parameterValue : configuration.getParameterValues())
            {
                if (std::get<0>(parameterValue) == constraintNames.at(i))
                {
                    constraintValues.at(i) = std::get<1>(parameterValue);
                    break;
                }
            }
        }

        auto constraintFunction = constraint.getConstraintFunction();
        if (!constraintFunction(constraintValues))
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt
