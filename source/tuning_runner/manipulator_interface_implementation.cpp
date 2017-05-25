#include <stdexcept>
#include <utility>

#include "manipulator_interface_implementation.h"
#include "../utility/ktt_utility.h"
#include "../utility/timer.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeApiDriver* computeApiDriver) :
    computeApiDriver(computeApiDriver),
    currentResult(KernelRunResult(0, 0, std::vector<KernelArgument> {})),
    currentConfiguration(KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>{})),
    automaticArgumentUpdate(false),
    synchronizeWriteArguments(true),
    synchronizeReadWriteArguments(true)
{}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    return runKernel(kernelId, dataPointer->second.getGlobalSize(), dataPointer->second.getLocalSize());
}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId, const DimensionVector& globalSize,
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
    currentResult = KernelRunResult(currentResult.getDuration() + result.getDuration(), currentResult.getOverhead(), result.getResultArguments());

    std::vector<ResultArgument> resultArguments;
    for (const auto& resultArgument : currentResult.getResultArguments())
    {
        resultArguments.emplace_back(ResultArgument(resultArgument.getId(), resultArgument.getData(), resultArgument.getNumberOfElements(),
            resultArgument.getElementSizeInBytes(), resultArgument.getArgumentDataType(), resultArgument.getArgumentMemoryType()));
    }

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());

    if (automaticArgumentUpdate)
    {
        for (const auto& resultArgument : resultArguments)
        {
            if (resultArgument.getArgumentMemoryType() == ArgumentMemoryType::WriteOnly && synchronizeWriteArguments
                || resultArgument.getArgumentMemoryType() == ArgumentMemoryType::ReadWrite && synchronizeReadWriteArguments)
            {
                updateArgumentVector(resultArgument.getId(), resultArgument.getData());
            }
        }
    }

    return resultArguments;
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
    updateArgument(argumentId, argumentData, 1, ArgumentUploadType::Scalar, false);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData)
{
    updateArgument(argumentId, argumentData, 0, ArgumentUploadType::Vector, false);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements)
{
    updateArgument(argumentId, argumentData, numberOfElements, ArgumentUploadType::Vector, true);
}

void ManipulatorInterfaceImplementation::setAutomaticArgumentUpdate(const bool flag)
{
    automaticArgumentUpdate = flag;
}

void ManipulatorInterfaceImplementation::setArgumentSynchronization(const bool flag, const ArgumentMemoryType& argumentMemoryType)
{
    if (argumentMemoryType == ArgumentMemoryType::WriteOnly)
    {
        synchronizeWriteArguments = flag;
    }
    if (argumentMemoryType == ArgumentMemoryType::ReadWrite)
    {
        synchronizeReadWriteArguments = flag;
    }

    if (flag)
    {
        computeApiDriver->clearCache(argumentMemoryType);
    }

    computeApiDriver->setCacheUsage(!flag, argumentMemoryType);
}

void ManipulatorInterfaceImplementation::updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId)
            + " was called inside tuning manipulator which did not advertise utilization of this kernel");
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
    this->kernelArguments = kernelArguments;
}

void ManipulatorInterfaceImplementation::clearData()
{
    kernelDataMap.clear();
    currentResult = KernelRunResult(0, 0, std::vector<KernelArgument>{});
    currentConfiguration = KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>{});
    automaticArgumentUpdate = false;
    synchronizeWriteArguments = true;
    synchronizeReadWriteArguments = true;
    computeApiDriver->setCacheUsage(true, ArgumentMemoryType::ReadOnly);
    computeApiDriver->setCacheUsage(false, ArgumentMemoryType::WriteOnly);
    computeApiDriver->setCacheUsage(false, ArgumentMemoryType::ReadWrite);
    computeApiDriver->clearCache();
}

KernelRunResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    return currentResult;
}

void ManipulatorInterfaceImplementation::updateArgument(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
    const ArgumentUploadType& argumentUploadType, const bool overrideNumberOfElements)
{
    for (auto& argument : kernelArguments)
    {
        if (argument.getId() == argumentId)
        {
            if (overrideNumberOfElements)
            {
                argument = KernelArgument(argumentId, argumentData, numberOfElements, argument.getArgumentDataType(),
                    argument.getArgumentMemoryType(), argumentUploadType);
            }
            else
            {
                argument = KernelArgument(argumentId, argumentData, argument.getNumberOfElements(), argument.getArgumentDataType(),
                    argument.getArgumentMemoryType(), argumentUploadType);
            }
            return;
        }
    }
}

std::vector<const KernelArgument*> ManipulatorInterfaceImplementation::getArgumentPointers(const std::vector<size_t>& argumentIndices)
{
    std::vector<const KernelArgument*> result;

    for (const auto index : argumentIndices)
    {
        for (const auto& argument : kernelArguments)
        {
            if (index == argument.getId())
            {
                result.push_back(&argument);
            }
        }
    }

    return result;
}

} // namespace ktt
