#include <iostream>
#include "tuner_api.h"
#include "tuner_core.h"

namespace ktt
{

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex) :
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex, ComputeApi::Opencl))
{}

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi) :
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex, computeApi))
{}

Tuner::~Tuner() = default;

KernelId Tuner::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return tunerCore->addKernel(source, kernelName, globalSize, localSize);
}

KernelId Tuner::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
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

void Tuner::setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    try
    {
        tunerCore->setKernelArguments(id, argumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
{
    try
    {
        tunerCore->addParameter(id, parameterName, parameterValues, ThreadModifierType::None, ThreadModifierAction::Multiply, Dimension::X);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
    const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addParameter(id, parameterName, parameterValues, modifierType, modifierAction, modifierDimension);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    try
    {
        tunerCore->addConstraint(id, constraintFunction, parameterNames);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    try
    {
        tunerCore->setTuningManipulator(id, std::move(manipulator));
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

KernelId Tuner::addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
    std::unique_ptr<TuningManipulator> manipulator)
{
    try
    {
        return tunerCore->addComposition(compositionName, kernelIds, std::move(manipulator));
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
    const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addCompositionKernelParameter(compositionId, kernelId, parameterName, parameterValues, modifierType, modifierAction,
            modifierDimension);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
{
    try
    {
        tunerCore->setCompositionKernelArguments(compositionId, kernelId, argumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::tuneKernel(const KernelId id)
{
    try
    {
        tunerCore->tuneKernel(id);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
{
    try
    {
        tunerCore->tuneKernelByStep(id, output);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    try
    {
        tunerCore->runKernel(id, configuration, output);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments)
{
    try
    {
        tunerCore->setSearchMethod(method, arguments);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setPrintingTimeUnit(const TimeUnit& unit)
{
    tunerCore->setPrintingTimeUnit(unit);
}

void Tuner::setInvalidResultPrinting(const bool flag)
{
    tunerCore->setInvalidResultPrinting(flag);
}

void Tuner::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const
{
    try
    {
        tunerCore->printResult(id, outputTarget, format);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const
{
    try
    {
        tunerCore->printResult(id, filePath, format);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

std::vector<ParameterPair> Tuner::getBestConfiguration(const KernelId id) const
{
    try
    {
        return tunerCore->getBestConfiguration(id);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    try
    {
        tunerCore->setReferenceKernel(id, referenceId, referenceConfiguration, validatedArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds)
{
    try
    {
        tunerCore->setReferenceClass(id, std::move(referenceClass), validatedArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setValidationMethod(const ValidationMethod& method, const double toleranceThreshold)
{
    try
    {
        tunerCore->setValidationMethod(method, toleranceThreshold);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setValidationRange(const ArgumentId id, const size_t range)
{
    try
    {
        tunerCore->setValidationRange(id, range);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
{
    try
    {
        tunerCore->setArgumentComparator(id, comparator);
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
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
        tunerCore->log(error.what());
        throw;
    }
}

DeviceInfo Tuner::getCurrentDeviceInfo() const
{
    try
    {
        return tunerCore->getCurrentDeviceInfo();
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setAutomaticGlobalSizeCorrection(const bool flag)
{
    tunerCore->setAutomaticGlobalSizeCorrection(flag);
}

void Tuner::setGlobalSizeType(const GlobalSizeType& type)
{
    tunerCore->setGlobalSizeType(type);
}

void Tuner::setLoggingTarget(std::ostream& outputTarget)
{
    tunerCore->setLoggingTarget(outputTarget);
}

void Tuner::setLoggingTarget(const std::string& filePath)
{
    tunerCore->setLoggingTarget(filePath);
}

ArgumentId Tuner::addArgument(void* vectorData, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const bool copyData)
{
    try
    {
        return tunerCore->addArgument(vectorData, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType,
            ArgumentUploadType::Vector, copyData);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

ArgumentId Tuner::addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType)
{
    try
    {
        return tunerCore->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

ArgumentId Tuner::addArgument(const size_t localMemoryElementsCount, const size_t elementSizeInBytes, const ArgumentDataType& dataType)
{
    try
    {
        return tunerCore->addArgument(nullptr, localMemoryElementsCount, elementSizeInBytes, dataType, ArgumentMemoryLocation::Device,
            ArgumentAccessType::ReadOnly, ArgumentUploadType::Local);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

} // namespace ktt
