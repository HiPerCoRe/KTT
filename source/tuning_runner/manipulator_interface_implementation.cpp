#include <cstring>
#include <stdexcept>
#include <utility>

#include "manipulator_interface_implementation.h"
#include "../utility/ktt_utility.h"

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
    auto dataPointer = kernelDataMap.find(kernelId);
    if (dataPointer == kernelDataMap.end())
    {
        throw std::runtime_error(std::string("Kernel with id: ") + std::to_string(kernelId) +
            " was called inside tuning manipulator which did not advertise utilization of this kernel");
    }
    KernelRuntimeData kernelData = dataPointer->second;

    KernelRunResult result = computeApiDriver->runKernel(kernelData.getSource(), kernelData.getName(), convertDimensionVector(globalSize),
        convertDimensionVector(localSize), getArguments(kernelData.getArgumentIndices()));
    currentResult = KernelRunResult(currentResult.getDuration() + result.getDuration(), currentResult.getOverhead() + result.getOverhead(),
        result.getResultArguments());

    std::vector<ResultArgument> resultArguments;
    for (const auto& resultArgument : currentResult.getResultArguments())
    {
        resultArguments.emplace_back(ResultArgument(kernelId, resultArgument.getData(), resultArgument.getDataSizeInBytes(),
            resultArgument.getArgumentDataType()));
    }

    return resultArguments;
}

void ManipulatorInterfaceImplementation::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    for (auto& argument : kernelArguments)
    {
        if (argument.getId() == argumentId)
        {
            if (argument.getArgumentDataType() == ArgumentDataType::Double)
            {
                std::vector<double> data;
                data.resize(1);
                std::memcpy(data.data(), argumentData, sizeof(double));
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Scalar);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Float)
            {
                std::vector<float> data;
                data.resize(1);
                std::memcpy(data.data(), argumentData, sizeof(float));
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Scalar);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Int)
            {
                std::vector<int> data;
                data.resize(1);
                std::memcpy(data.data(), argumentData, sizeof(int));
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Scalar);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Short)
            {
                std::vector<short> data;
                data.resize(1);
                std::memcpy(data.data(), argumentData, sizeof(short));
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Scalar);
            }
            else
            {
                throw std::runtime_error("Unsupported argument data type");
            }
            return;
        }
    }
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes)
{
    for (auto& argument : kernelArguments)
    {
        if (argument.getId() == argumentId)
        {
            if (argument.getArgumentDataType() == ArgumentDataType::Double)
            {
                std::vector<double> data;
                data.resize(dataSizeInBytes / sizeof(double));
                std::memcpy(data.data(), argumentData, dataSizeInBytes);
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Vector);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Float)
            {
                std::vector<float> data;
                data.resize(dataSizeInBytes / sizeof(float));
                std::memcpy(data.data(), argumentData, dataSizeInBytes);
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Vector);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Int)
            {
                std::vector<int> data;
                data.resize(dataSizeInBytes / sizeof(int));
                std::memcpy(data.data(), argumentData, dataSizeInBytes);
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Vector);
            }
            else if (argument.getArgumentDataType() == ArgumentDataType::Short)
            {
                std::vector<short> data;
                data.resize(dataSizeInBytes / sizeof(short));
                std::memcpy(data.data(), argumentData, dataSizeInBytes);
                argument = KernelArgument(argumentId, data, argument.getArgumentMemoryType(), ArgumentQuantity::Vector);
            }
            else
            {
                throw std::runtime_error("Unsupported argument data type");
            }
            return;
        }
    }
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
