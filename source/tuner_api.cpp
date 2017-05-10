#include <iostream>

#include "tuner_api.h"
#include "tuner_core.h"

namespace ktt
{

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex) :
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    try
    {
        tunerCore->setTuningManipulator(kernelId, std::move(tuningManipulator));
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
        throw;
    }
}

template size_t Tuner::addArgument<short>(const std::vector<short>& data, const ArgumentMemoryType& argumentMemoryType);
template size_t Tuner::addArgument<int>(const std::vector<int>& data, const ArgumentMemoryType& argumentMemoryType);
template size_t Tuner::addArgument<float>(const std::vector<float>& data, const ArgumentMemoryType& argumentMemoryType);
template size_t Tuner::addArgument<double>(const std::vector<double>& data, const ArgumentMemoryType& argumentMemoryType);

template <typename T> size_t Tuner::addArgument(const T value)
{
    try
    {
        return tunerCore->addArgument(std::vector<T>{ value }, ArgumentMemoryType::ReadOnly, ArgumentQuantity::Scalar);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

template size_t Tuner::addArgument<short>(const short value);
template size_t Tuner::addArgument<int>(const int value);
template size_t Tuner::addArgument<float>(const float value);
template size_t Tuner::addArgument<double>(const double value);

void Tuner::enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)
{
    try
    {
        tunerCore->enableArgumentPrinting(argumentId, filePath, argumentPrintCondition);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::tuneKernel(const size_t kernelId)
{
    try
    {
        tunerCore->tuneKernel(kernelId);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
    }
}

void Tuner::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    try
    {
        tunerCore->setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds)
{
    try
    {
        tunerCore->setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    try
    {
        tunerCore->setValidationMethod(validationMethod, toleranceThreshold);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

void Tuner::printComputeApiInfo(std::ostream& outputTarget) const
{
    try
    {
        tunerCore->printComputeApiInfo(outputTarget);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

std::vector<PlatformInfo> Tuner::getPlatformInfo() const
{
    try
    {
        return tunerCore->getPlatformInfo();
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

std::vector<DeviceInfo> Tuner::getDeviceInfo(const size_t platformIndex) const
{
    try
    {
        return tunerCore->getDeviceInfo(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        throw;
    }
}

void Tuner::setLoggingTarget(std::ostream& outputTarget)
{
    tunerCore->setLoggingTarget(outputTarget);
}

void Tuner::setLoggingTarget(const std::string& filePath)
{
    tunerCore->setLoggingTarget(filePath);
}

} // namespace ktt
