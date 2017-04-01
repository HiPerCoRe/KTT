#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager():
    kernelCount(0)
{}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    kernels.emplace_back(Kernel(source, kernelName, globalSize, localSize));
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
    std::string source = getKernel(id).getSource();

    for (const auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterValue); // clean way to convert number to string
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
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
        computeConfigurations(0, kernels.at(id).getParameters(), std::vector<ParameterValue>(0), kernels.at(id).getGlobalSize(),
            kernels.at(id).getLocalSize(), configurations);
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

size_t KernelManager::getKernelCount() const
{
    return kernelCount;
}

const Kernel KernelManager::getKernel(const size_t id) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    return kernels.at(id);
}

std::string KernelManager::loadFileToString(const std::string& filePath) const
{
    std::ifstream file(filePath);
    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

void KernelManager::computeConfigurations(const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
    const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize, const DimensionVector& localSize,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterValues);
        finalResult.push_back(configuration);
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterValues = parameterValues;
        newParameterValues.push_back(ParameterValue(parameter.getName(), value));

        auto newGlobalSize = modifyDimensionVector(globalSize, DimensionVectorType::Global, parameter, value);
        auto newLocalSize = modifyDimensionVector(localSize, DimensionVectorType::Local, parameter, value);

        computeConfigurations(currentParameterIndex + 1, parameters, newParameterValues, newGlobalSize, newLocalSize, finalResult);
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

} // namespace ktt
