#include <stdexcept>
#include <utility>

#include "manipulator_interface_implementation.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeEngine* computeEngine) :
    computeEngine(computeEngine),
    currentResult(KernelRunResult(0, 0)),
    currentConfiguration(KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>{}))
{}

void ManipulatorInterfaceImplementation::runKernel(const size_t kernelId)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    runKernel(kernelId, dataPointer->second.getGlobalSizeDimensionVector(), dataPointer->second.getLocalSizeDimensionVector());
}

void ManipulatorInterfaceImplementation::runKernel(const size_t kernelId, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    Timer timer;
    timer.start();

    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    KernelRuntimeData kernelData = dataPointer->second;

    KernelRunResult result = computeEngine->runKernel(kernelData, getArgumentPointers(kernelData.getArgumentIndices()), {});
    currentResult = KernelRunResult(currentResult.getDuration() + result.getDuration(), currentResult.getOverhead());

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
}

DimensionVector ManipulatorInterfaceImplementation::getCurrentGlobalSize(const size_t kernelId) const
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    return dataPointer->second.getGlobalSizeDimensionVector();
}

DimensionVector ManipulatorInterfaceImplementation::getCurrentLocalSize(const size_t kernelId) const
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    return dataPointer->second.getLocalSizeDimensionVector();
}

std::vector<ParameterValue> ManipulatorInterfaceImplementation::getCurrentConfiguration() const
{
    return currentConfiguration.getParameterValues();
}

void ManipulatorInterfaceImplementation::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    updateArgumentHost(argumentId, argumentData, 1, ArgumentUploadType::Scalar);
}

void ManipulatorInterfaceImplementation::updateArgumentLocal(const size_t argumentId, const size_t numberOfElements)
{
    updateArgumentHost(argumentId, nullptr, numberOfElements, ArgumentUploadType::Local);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData)
{
    auto argumentPointer = vectorArgumentMap.find(argumentId);
    if (argumentPointer == vectorArgumentMap.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
    }

    computeEngine->updateArgument(argumentId, argumentData, argumentPointer->second->getDataSizeInBytes());
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements)
{
    auto argumentPointer = vectorArgumentMap.find(argumentId);
    if (argumentPointer == vectorArgumentMap.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
    }

    computeEngine->updateArgument(argumentId, argumentData, argumentPointer->second->getElementSizeInBytes() * numberOfElements);
}

void ManipulatorInterfaceImplementation::getArgumentVector(const size_t argumentId, void* destination) const
{
    computeEngine->downloadArgument(argumentId, destination);
}

void ManipulatorInterfaceImplementation::getArgumentVector(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const
{
    computeEngine->downloadArgument(argumentId, destination, dataSizeInBytes);
}

void ManipulatorInterfaceImplementation::changeKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }

    if (!containsUnique(argumentIds))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    dataPointer->second.setArgumentIndices(argumentIds);
}

void ManipulatorInterfaceImplementation::swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }

    auto indices = dataPointer->second.getArgumentIndices();
    
    if (!elementExists(argumentIdFirst, indices) || !elementExists(argumentIdSecond, indices))
    {
        throw std::runtime_error(std::string("One of the following argument ids are not associated with this kernel: ")
            + std::to_string(argumentIdFirst) + ", " + std::to_string(argumentIdSecond) + ", kernel id: " + std::to_string(kernelId));
    }

    size_t firstIndex;
    size_t secondIndex;
    for (size_t i = 0; i < indices.size(); i++)
    {
        if (indices.at(i) == argumentIdFirst)
        {
            firstIndex = i;
        }
        if (indices.at(i) == argumentIdSecond)
        {
            secondIndex = i;
        }
    }
    std::swap(indices.at(firstIndex), indices.at(secondIndex));

    dataPointer->second.setArgumentIndices(indices);
}

void ManipulatorInterfaceImplementation::addKernel(const size_t id, const KernelRuntimeData& kernelRuntimeData)
{
    kernelDataMap.insert(std::make_pair(id, kernelRuntimeData));
}

void ManipulatorInterfaceImplementation::setConfiguration(const KernelConfiguration& kernelConfiguration)
{
    currentConfiguration = kernelConfiguration;
}

void ManipulatorInterfaceImplementation::setKernelArguments(const std::vector<KernelArgument*>& kernelArguments)
{
    for (const auto& kernelArgument : kernelArguments)
    {
        if (kernelArgument->getArgumentUploadType() == ArgumentUploadType::Vector)
        {
            vectorArgumentMap.insert(std::make_pair(kernelArgument->getId(), kernelArgument));
        }
        else
        {
            nonVectorArgumentMap.insert(std::make_pair(kernelArgument->getId(), *kernelArgument));
        }
    }
}

void ManipulatorInterfaceImplementation::uploadBuffers()
{
    for (auto& argument : vectorArgumentMap)
    {
        computeEngine->uploadArgument(*argument.second);
    }
}

void ManipulatorInterfaceImplementation::downloadBuffers(const std::vector<ArgumentOutputDescriptor>& outputDescriptors) const
{
    for (const auto& descriptor : outputDescriptors)
    {
        if (descriptor.getOutputSizeInBytes() == 0)
        {
            computeEngine->downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination());
        }
        else
        {
            computeEngine->downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
        }
    }
}

void ManipulatorInterfaceImplementation::clearData()
{
    kernelDataMap.clear();
    currentResult = KernelRunResult(0, 0);
    currentConfiguration = KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>{});
    vectorArgumentMap.clear();
    nonVectorArgumentMap.clear();
}

KernelRunResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    return currentResult;
}

std::vector<KernelArgument*> ManipulatorInterfaceImplementation::getArgumentPointers(const std::vector<size_t>& argumentIndices)
{
    std::vector<KernelArgument*> result;

    for (const auto index : argumentIndices)
    {
        bool argumentAdded = false;

        for (const auto argument : vectorArgumentMap)
        {
            if (index == argument.second->getId())
            {
                result.push_back(argument.second);
                argumentAdded = true;
                break;
            }
        }

        if (argumentAdded)
        {
            continue;
        }

        for (auto& argument : nonVectorArgumentMap)
        {
            if (index == argument.second.getId())
            {
                result.push_back(&argument.second);
                break;
            }
        }
    }

    return result;
}

void ManipulatorInterfaceImplementation::updateArgumentHost(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
    const ArgumentUploadType& argumentUploadType)
{
    auto argumentPointer = nonVectorArgumentMap.find(argumentId);
    if (argumentPointer == nonVectorArgumentMap.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
    }

    if (argumentPointer->second.getArgumentUploadType() != argumentUploadType)
    {
        throw std::runtime_error("Cannot convert between scalar and vector arguments");
    }

    auto updatedArgument = KernelArgument(argumentId, argumentData, numberOfElements, argumentPointer->second.getArgumentDataType(),
        argumentPointer->second.getArgumentMemoryLocation(), argumentPointer->second.getArgumentAccessType(), argumentUploadType);

    nonVectorArgumentMap.erase(argumentId);
    nonVectorArgumentMap.insert(std::make_pair(argumentId, updatedArgument));
}

} // namespace ktt
