#include <stdexcept>
#include <utility>

#include "manipulator_interface_implementation.h"
#include "../utility/ktt_utility.h"
#include "../utility/timer.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeApiDriver* computeApiDriver) :
    computeApiDriver(computeApiDriver),
    currentResult(KernelRunResult(0, 0, std::vector<KernelArgument> {}))
{}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId)
{
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId) +
            " was called inside tuning manipulator which did not advertise utilization of this kernel");
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
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId) +
            " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    KernelRuntimeData kernelData = dataPointer->second;

    KernelRunResult result = computeApiDriver->runKernel(kernelData.getSource(), kernelData.getName(), convertDimensionVector(globalSize),
        convertDimensionVector(localSize), getArguments(kernelData.getArgumentIndices()));
    currentResult = KernelRunResult(currentResult.getDuration() + result.getDuration(), currentResult.getOverhead(), result.getResultArguments());

    std::vector<ResultArgument> resultArguments;
    for (const auto& resultArgument : currentResult.getResultArguments())
    {
        resultArguments.emplace_back(ResultArgument(resultArgument.getId(), resultArgument.getData(), resultArgument.getNumberOfElements(),
            resultArgument.getElementSizeInBytes(), resultArgument.getArgumentDataType()));
    }

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
    return resultArguments;
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

void ManipulatorInterfaceImplementation::addKernel(const size_t id, const KernelRuntimeData& kernelRuntimeData)
{
    kernelDataMap.insert(std::make_pair(id, kernelRuntimeData));
}

void ManipulatorInterfaceImplementation::setKernelArguments(const std::vector<KernelArgument>& kernelArguments)
{
    this->kernelArguments = kernelArguments;
}

void ManipulatorInterfaceImplementation::clearData()
{
    kernelDataMap.clear();
    currentResult = KernelRunResult(0, 0, std::vector<KernelArgument>{});
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

std::vector<KernelArgument> ManipulatorInterfaceImplementation::getArguments(const std::vector<size_t>& argumentIndices)
{
    std::vector<KernelArgument> result;

    for (const auto index : argumentIndices)
    {
        for (const auto& argument : kernelArguments)
        {
            if (index == argument.getId())
            {
                result.push_back(argument);
            }
        }
    }

    return result;
}

} // namespace ktt
