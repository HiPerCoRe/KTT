#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager() :
    nextId(0),
    globalSizeType(GlobalSizeType::Opencl)
{}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    DimensionVector convertedGlobalSize = globalSize;

    if (globalSizeType == GlobalSizeType::Cuda)
    {
        convertedGlobalSize = DimensionVector(std::get<0>(globalSize) * std::get<0>(localSize), std::get<1>(globalSize) * std::get<1>(localSize),
            std::get<2>(globalSize) * std::get<2>(localSize));
    }

    kernels.emplace_back(nextId, source, kernelName, convertedGlobalSize, localSize);
    return nextId++;
}

size_t KernelManager::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filePath);
    return addKernel(source, kernelName, globalSize, localSize);
}

size_t KernelManager::addKernelComposition(const std::vector<size_t> kernelIds)
{
    std::vector<const Kernel*> compositionKernels;
    for (const auto& id : kernelIds)
    {
        if (!isKernel(id))
        {
            throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
        }
        compositionKernels.push_back(&kernels.at(id));
    }

    kernelCompositions.emplace_back(nextId, compositionKernels);
    return nextId++;
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

KernelConfiguration KernelManager::getKernelConfiguration(const size_t id, const std::vector<ParameterValue>& parameterValues) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = getKernel(id);
    DimensionVector global = kernel.getGlobalSize();
    DimensionVector local = kernel.getLocalSize();
    
    for (const auto& value : parameterValues)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernel.getParameters())
        {
            if (parameter.getName() == std::get<0>(value))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <") + std::get<0>(value) + "> is not associated with kernel with id: "
                + std::to_string(id));
        }

        global = modifyDimensionVector(global, DimensionVectorType::Global, *targetParameter, std::get<1>(value));
        local = modifyDimensionVector(local, DimensionVectorType::Local, *targetParameter, std::get<1>(value));
    }

    return KernelConfiguration(global, local, parameterValues);
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurations(const size_t id, const DeviceInfo& deviceInfo) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<KernelConfiguration> configurations;
    const Kernel& kernel = getKernel(id);

    if (kernel.getParameters().size() == 0)
    {
        configurations.emplace_back(kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<ParameterValue>{});
    }
    else
    {
        computeConfigurations(0, deviceInfo, kernel.getParameters(), kernel.getConstraints(), std::vector<ParameterValue>(0), kernel.getGlobalSize(),
            kernel.getLocalSize(), configurations);
    }
    return configurations;
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurationsWithComposition(const size_t kernelId, const size_t compositionId,
    const DeviceInfo& deviceInfo) const
{
    if (!isKernel(kernelId))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }

    if (!isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid composition id: ") + std::to_string(compositionId));
    }

    std::vector<KernelConfiguration> configurations;
    const Kernel& kernel = getKernel(kernelId);
    const KernelComposition& composition = getKernelComposition(compositionId);

    std::vector<KernelParameter> allParameters = kernel.getParameters();
    std::vector<KernelParameter> compositionParameters = composition.getParameters();
    allParameters.insert(std::end(allParameters), std::begin(compositionParameters), std::end(compositionParameters));

    if (allParameters.size() == 0)
    {
        configurations.emplace_back(kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<ParameterValue>{});
    }
    else
    {
        std::vector<KernelConstraint> allConstraints = kernel.getConstraints();
        std::vector<KernelConstraint> compositionConstraints = composition.getConstraintsForKernel(kernelId);
        allConstraints.insert(std::end(allConstraints), std::begin(compositionConstraints), std::end(compositionConstraints));

        computeConfigurations(0, deviceInfo, allParameters, allConstraints, std::vector<ParameterValue>(0), kernel.getGlobalSize(),
            kernel.getLocalSize(), configurations);
    }
    return configurations;
}

void KernelManager::setGlobalSizeType(const GlobalSizeType& globalSizeType)
{
    this->globalSizeType = globalSizeType;
}

void KernelManager::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    if (isKernel(id))
    {
        getKernel(id).addParameter(KernelParameter(name, values, threadModifierType, threadModifierAction, modifierDimension));
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).addParameter(KernelParameter(name, values, threadModifierType, threadModifierAction, modifierDimension));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    if (isKernel(id))
    {
        getKernel(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::setArguments(const size_t id, const std::vector<size_t>& argumentIndices)
{
    if (isKernel(id))
    {
        getKernel(id).setArguments(argumentIndices);
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).setArguments(argumentIndices);
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

size_t KernelManager::getKernelCount() const
{
    return kernels.size();
}

const Kernel& KernelManager::getKernel(const size_t id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return kernel;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

Kernel& KernelManager::getKernel(const size_t id)
{
    return const_cast<Kernel&>(static_cast<const KernelManager*>(this)->getKernel(id));
}

size_t KernelManager::getCompositionCount() const
{
    return kernelCompositions.size();
}

const KernelComposition& KernelManager::getKernelComposition(const size_t id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return kernelComposition;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
}

KernelComposition& KernelManager::getKernelComposition(const size_t id)
{
    return const_cast<KernelComposition&>(static_cast<const KernelManager*>(this)->getKernelComposition(id));
}

bool KernelManager::isKernel(const size_t id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return true;
        }
    }

    return false;
}

bool KernelManager::isKernelComposition(const size_t id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return true;
        }
    }

    return false;
}

std::string KernelManager::loadFileToString(const std::string& filePath) const
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

void KernelManager::computeConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize, const DimensionVector& localSize,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterValues);
        if (configurationIsValid(configuration, constraints, deviceInfo))
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

        computeConfigurations(currentParameterIndex + 1, deviceInfo, parameters, constraints, newParameterValues, newGlobalSize, newLocalSize,
            finalResult);
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

bool KernelManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints,
    const DeviceInfo& deviceInfo) const
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

    auto localSize = configuration.getLocalSize();
    if (std::get<0>(localSize) * std::get<1>(localSize) * std::get<2>(localSize) > deviceInfo.getMaxWorkGroupSize())
    {
        return false;
    }

    return true;
}

} // namespace ktt
