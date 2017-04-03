#include <iostream>

#include "tuner_api.h"
#include "tuner_core.h"

namespace ktt
{

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex):
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex))
{}

Tuner::~Tuner() = default;

size_t Tuner::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return tunerCore->addKernel(source, kernelName, globalSize, localSize);
}

size_t Tuner::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    try
    {
        return tunerCore->addKernelFromFile(filePath, kernelName, globalSize, localSize);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)
{
    try
    {
        tunerCore->addParameter(kernelId, name, values, ThreadModifierType::None, ThreadModifierAction::Multiply, Dimension::X);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addParameter(kernelId, name, values, threadModifierType, threadModifierAction, modifierDimension);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    try
    {
        tunerCore->addConstraint(kernelId, constraintFunction, parameterNames);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIndices)
{
    try
    {
        tunerCore->setKernelArguments(kernelId, argumentIndices);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    try
    {
        tunerCore->setSearchMethod(kernelId, searchMethod, searchArguments);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

template <typename T> size_t Tuner::addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)
{
    try
    {
        return tunerCore->addArgument(data, argumentMemoryType, ArgumentQuantity::Vector);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

template size_t Tuner::addArgument<int>(const std::vector<int>& data, const ArgumentMemoryType& argumentMemoryType);
template size_t Tuner::addArgument<float>(const std::vector<float>& data, const ArgumentMemoryType& argumentMemoryType);
template size_t Tuner::addArgument<double>(const std::vector<double>& data, const ArgumentMemoryType& argumentMemoryType);

template <typename T> size_t Tuner::addArgument(const T value)
{
    try
    {
        return tunerCore->addArgument(std::vector<T>{ value }, ArgumentMemoryType::ReadWrite, ArgumentQuantity::Scalar);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

template size_t Tuner::addArgument<int>(const int value);
template size_t Tuner::addArgument<float>(const float value);
template size_t Tuner::addArgument<double>(const double value);

template <typename T> void Tuner::updateArgument(const size_t argumentId, const std::vector<T>& data)
{
    try
    {
        tunerCore->updateArgument(argumentId, data, ArgumentQuantity::Vector);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

template void Tuner::updateArgument<int>(const size_t argumentId, const std::vector<int>& data);
template void Tuner::updateArgument<float>(const size_t argumentId, const std::vector<float>& data);
template void Tuner::updateArgument<double>(const size_t argumentId, const std::vector<double>& data);

template <typename T> void Tuner::updateArgument(const size_t argumentId, const T value)
{
    try
    {
        tunerCore->updateArgument(argumentId, std::vector<T>{ value }, ArgumentQuantity::Scalar);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

template void Tuner::updateArgument<int>(const size_t argumentId, const int value);
template void Tuner::updateArgument<float>(const size_t argumentId, const float value);
template void Tuner::updateArgument<double>(const size_t argumentId, const double value);

void Tuner::tuneKernel(const size_t kernelId)
{
    try
    {
        tunerCore->tuneKernel(kernelId);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
{
    try
    {
        tunerCore->printResult(kernelId, outputTarget, printFormat);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const
{
    try
    {
        tunerCore->printResult(kernelId, filePath, printFormat);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

void Tuner::printComputeAPIInfo(std::ostream& outputTarget)
{
    try
    {
        TunerCore::printComputeAPIInfo(outputTarget);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

PlatformInfo Tuner::getPlatformInfo(const size_t platformIndex)
{
    try
    {
        return TunerCore::getPlatformInfo(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

std::vector<PlatformInfo> Tuner::getPlatformInfoAll()
{
    try
    {
        return TunerCore::getPlatformInfoAll();
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

DeviceInfo Tuner::getDeviceInfo(const size_t platformIndex, const size_t deviceIndex)
{
    try
    {
        return TunerCore::getDeviceInfo(platformIndex, deviceIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

std::vector<DeviceInfo> Tuner::getDeviceInfoAll(const size_t platformIndex)
{
    try
    {
        return TunerCore::getDeviceInfoAll(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

} // namespace ktt
