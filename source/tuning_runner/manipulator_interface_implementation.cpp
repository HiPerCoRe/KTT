#include <stdexcept>
#include <utility>

#include "manipulator_interface_implementation.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeApiDriver* computeApiDriver) :
    computeApiDriver(computeApiDriver),
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
    runKernel(kernelId, dataPointer->second.getGlobalSize(), dataPointer->second.getLocalSize());
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

    KernelRunResult result = computeApiDriver->runKernel(kernelData.getSource(), kernelData.getName(), convertDimensionVector(globalSize),
        convertDimensionVector(localSize), getArgumentPointers(kernelData.getArgumentIndices()));
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
    return dataPointer->second.getGlobalSize();
}

DimensionVector ManipulatorInterfaceImplementation::getCurrentLocalSize(const size_t kernelId) const
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    return dataPointer->second.getLocalSize();
}

std::vector<ParameterValue> ManipulatorInterfaceImplementation::getCurrentConfiguration() const
{
    return currentConfiguration.getParameterValues();
}

void ManipulatorInterfaceImplementation::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    updateArgumentHost(argumentId, argumentData, 1, ArgumentUploadType::Scalar);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData,
    const ArgumentLocation& argumentLocation)
{
    auto argumentPointer = kernelArgumentMap.find(argumentId);
    if (argumentPointer == kernelArgumentMap.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
    }

    updateArgumentVector(argumentId, argumentData, argumentLocation, argumentPointer->second.getNumberOfElements());
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData,
    const ArgumentLocation& argumentLocation, const size_t numberOfElements)
{
    if (argumentLocation == ArgumentLocation::Host || argumentLocation == ArgumentLocation::HostAndDevice)
    {
        updateArgumentHost(argumentId, argumentData, numberOfElements, ArgumentUploadType::Vector);
    }

    if (argumentLocation == ArgumentLocation::Device || argumentLocation == ArgumentLocation::HostAndDevice)
    {
        auto argumentPointer = kernelArgumentMap.find(argumentId);
        if (argumentPointer == kernelArgumentMap.end())
        {
            throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
        }

        updateArgumentDevice(argumentId, argumentData, argumentPointer->second.getElementSizeInBytes() * numberOfElements);
    }
}

void ManipulatorInterfaceImplementation::synchronizeArgumentVector(const size_t argumentId, const bool downloadToHost)
{
    if (downloadToHost)
    {
        KernelArgument deviceArgument = computeApiDriver->downloadArgument(argumentId);
        updateArgumentVector(argumentId, deviceArgument.getData(), ArgumentLocation::Host);
    }
    else
    {
        auto argumentPointer = kernelArgumentMap.find(argumentId);
        if (argumentPointer == kernelArgumentMap.end())
        {
            throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
        }

        updateArgumentDevice(argumentId, argumentPointer->second.getData(), argumentPointer->second.getDataSizeInBytes());
    }
}

void ManipulatorInterfaceImplementation::setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)
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

void ManipulatorInterfaceImplementation::setKernelArguments(const std::vector<KernelArgument>& kernelArguments)
{
    for (const auto& kernelArgument : kernelArguments)
    {
        kernelArgumentMap.insert(std::make_pair(kernelArgument.getId(), kernelArgument));
    }
}

void ManipulatorInterfaceImplementation::clearData()
{
    kernelDataMap.clear();
    currentResult = KernelRunResult(0, 0);
    currentConfiguration = KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>{});
    kernelArgumentMap.clear();
}

KernelRunResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    return currentResult;
}

std::vector<const KernelArgument*> ManipulatorInterfaceImplementation::getArgumentPointers(const std::vector<size_t>& argumentIndices)
{
    std::vector<const KernelArgument*> result;

    for (const auto index : argumentIndices)
    {
        for (const auto& argument : kernelArgumentMap)
        {
            if (index == argument.second.getId())
            {
                result.push_back(&argument.second);
            }
        }
    }

    return result;
}

void ManipulatorInterfaceImplementation::updateArgumentHost(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
    const ArgumentUploadType& argumentUploadType)
{
    auto argumentPointer = kernelArgumentMap.find(argumentId);
    if (argumentPointer == kernelArgumentMap.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(argumentId));
    }

    if (argumentPointer->second.getArgumentUploadType() != argumentUploadType)
    {
        throw std::runtime_error("Cannot convert between scalar and vector arguments");
    }

    auto updatedArgument = KernelArgument(argumentId, argumentData, numberOfElements, argumentPointer->second.getArgumentDataType(),
        argumentPointer->second.getArgumentMemoryType(), argumentUploadType);

    kernelArgumentMap.erase(argumentId);
    kernelArgumentMap.insert(std::make_pair(argumentId, updatedArgument));
}

void ManipulatorInterfaceImplementation::updateArgumentDevice(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes)
{
    computeApiDriver->updateArgument(argumentId, argumentData, dataSizeInBytes);
}

} // namespace ktt
