#include <iostream>

#include "tuner_api.h"
#include "tuner_core.h"

namespace ktt
{

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex):
    tunerCore(std::make_unique<TunerCore>(platformIndex, std::vector<size_t>{ deviceIndex }))
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

void Tuner::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values)
{
    try
    {
        tunerCore->addParameter(id, name, values, ThreadModifierType::None, Dimension::X);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
    const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addParameter(id, name, values, threadModifierType, modifierDimension);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentInt(const size_t id, const std::vector<int>& data, const ArgumentMemoryType& argumentMemoryType)
{
    try
    {
        tunerCore->addArgumentInt(id, data, argumentMemoryType);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentFloat(const size_t id, const std::vector<float>& data, const ArgumentMemoryType& argumentMemoryType)
{
    try
    {
        tunerCore->addArgumentFloat(id, data, argumentMemoryType);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentDouble(const size_t id, const std::vector<double>& data, const ArgumentMemoryType& argumentMemoryType)
{
    try
    {
        tunerCore->addArgumentDouble(id, data, argumentMemoryType);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    try
    {
        tunerCore->useSearchMethod(id, searchMethod, searchArguments);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
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
    }
}

std::vector<Platform> Tuner::getPlatformInfo()
{
    std::vector<Platform> platformInfo;

    try
    {
        platformInfo = TunerCore::getPlatformInfo();
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }

    return platformInfo;
}

std::vector<Device> Tuner::getDeviceInfo(const size_t platformIndex)
{
    std::vector<Device> deviceInfo;

    try
    {
        deviceInfo = TunerCore::getDeviceInfo(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }

    return deviceInfo;
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

} // namespace ktt
