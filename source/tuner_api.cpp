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
    return tunerCore->addKernelFromFile(filePath, kernelName, globalSize, localSize);
}

void Tuner::addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)
{
    try
    {
        tunerCore->addParameter(kernelId, name, values, ThreadModifierType::None, Dimension::X);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addParameter(kernelId, name, values, threadModifierType, modifierDimension);
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

void Tuner::useSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    try
    {
        tunerCore->useSearchMethod(kernelId, searchMethod, searchArguments);
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
        return tunerCore->addArgument(data, argumentMemoryType);
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

template <typename T> void Tuner::updateArgument(const size_t argumentId, const std::vector<T>& data)
{
    try
    {
        tunerCore->updateArgument(argumentId, data);
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
    PlatformInfo platformInfo(0, std::string(""));

    try
    {
        platformInfo = TunerCore::getPlatformInfo(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }

    return platformInfo;
}

std::vector<PlatformInfo> Tuner::getPlatformInfoAll()
{
    std::vector<PlatformInfo> platformInfo;

    try
    {
        platformInfo = TunerCore::getPlatformInfoAll();
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }

    return platformInfo;
}

DeviceInfo Tuner::getDeviceInfo(const size_t platformIndex, const size_t deviceIndex)
{
    DeviceInfo deviceInfo(0, std::string(""));

    try
    {
        deviceInfo = TunerCore::getDeviceInfo(platformIndex, deviceIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }

    return deviceInfo;
}

std::vector<DeviceInfo> Tuner::getDeviceInfoAll(const size_t platformIndex)
{
    std::vector<DeviceInfo> deviceInfo;

    try
    {
        deviceInfo = TunerCore::getDeviceInfoAll(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }

    return deviceInfo;
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

} // namespace ktt
