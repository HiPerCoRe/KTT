#include <cstring>

#include "manipulator_interface_implementation.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ArgumentManager* argumentManager, OpenCLCore* openCLCore) :
    argumentManager(argumentManager),
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
    KernelRunResult result = openCLCore->runKernel(source, kernelName, convertDimensionVector(globalSize), convertDimensionVector(localSize),
        arguments);
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

void ManipulatorInterfaceImplementation::updateArgumentScalar(const size_t argumentId, const void* argumentData,
    const ArgumentDataType& argumentDataType)
{
    if (argumentDataType == ArgumentDataType::Double)
    {
        std::vector<double> data;
        data.resize(1);
        std::memcpy(data.data(), argumentData, sizeof(double));
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Scalar);
    }
    else if (argumentDataType == ArgumentDataType::Float)
    {
        std::vector<float> data;
        data.resize(1);
        std::memcpy(data.data(), argumentData, sizeof(float));
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Scalar);
    }
    else if (argumentDataType == ArgumentDataType::Int)
    {
        std::vector<int> data;
        data.resize(1);
        std::memcpy(data.data(), argumentData, sizeof(int));
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Scalar);
    }
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData,
    const ArgumentDataType& argumentDataType, const size_t dataSizeInBytes)
{
    if (argumentDataType == ArgumentDataType::Double)
    {
        std::vector<double> data;
        data.resize(dataSizeInBytes / sizeof(double));
        std::memcpy(data.data(), argumentData, dataSizeInBytes);
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Vector);
    }
    else if (argumentDataType == ArgumentDataType::Float)
    {
        std::vector<float> data;
        data.resize(dataSizeInBytes / sizeof(float));
        std::memcpy(data.data(), argumentData, dataSizeInBytes);
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Vector);
    }
    else if (argumentDataType == ArgumentDataType::Int)
    {
        std::vector<int> data;
        data.resize(dataSizeInBytes / sizeof(int));
        std::memcpy(data.data(), argumentData, dataSizeInBytes);
        argumentManager->updateArgument(argumentId, data, ArgumentQuantity::Vector);
    }
}

void ManipulatorInterfaceImplementation::setupKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<KernelArgument>& arguments)
{
    this->source = source;
    this->kernelName = kernelName;
    this->globalSize = globalSize;
    this->localSize = localSize;
    this->arguments = arguments;
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
