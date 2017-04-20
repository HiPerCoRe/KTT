#include <cstring>
#include <stdexcept>

#include "manipulator_interface_implementation.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(OpenCLCore* openCLCore) :
    openCLCore(openCLCore),
    currentResult(KernelRunResult(0, 0, std::vector<KernelArgument> {}))
{}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId)
{
    return runKernel(kernelId, globalSize, localSize);
}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    KernelRunResult result = openCLCore->runKernel(kernelSource, kernelName, convertDimensionVector(globalSize), convertDimensionVector(localSize),
        kernelArguments);
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

void ManipulatorInterfaceImplementation::setupKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<KernelArgument>& arguments)
{
    this->kernelSource = source;
    this->kernelName = kernelName;
    this->globalSize = globalSize;
    this->localSize = localSize;
    this->kernelArguments = arguments;
}

void ManipulatorInterfaceImplementation::resetCurrentResult()
{
    currentResult = KernelRunResult(0, 0, std::vector<KernelArgument> {});
}

KernelRunResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    return currentResult;
}

std::vector<size_t> ManipulatorInterfaceImplementation::convertDimensionVector(const DimensionVector& vector) const
{
    std::vector<size_t> result;

    result.push_back(std::get<0>(vector));
    result.push_back(std::get<1>(vector));
    result.push_back(std::get<2>(vector));

    return result;
}

} // namespace ktt
