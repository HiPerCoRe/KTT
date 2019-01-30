#include <iostream>
#include <tuner_api.h>
#include <tuner_core.h>

namespace ktt
{

Tuner::Tuner(const PlatformIndex platform, const DeviceIndex device) :
    tunerCore(std::make_unique<TunerCore>(platform, device, ComputeAPI::OpenCL, 1))
{}

Tuner::Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI) :
    tunerCore(std::make_unique<TunerCore>(platform, device, computeAPI, 1))
{}

Tuner::Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI, const uint32_t computeQueueCount) :
    tunerCore(std::make_unique<TunerCore>(platform, device, computeAPI, computeQueueCount))
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
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
{
    try
    {
        tunerCore->addParameter(id, parameterName, parameterValues);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::addParameterDouble(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues)
{
    try
    {
        tunerCore->addParameter(id, parameterName, parameterValues);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::addParameterPack(const KernelId id, const std::string& packName, const std::vector<std::string>& parameterNames)
{
    try
    {
        tunerCore->addParameterPack(id, packName, parameterNames);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    try
    {
        tunerCore->setThreadModifier(id, modifierType, modifierDimension, parameterNames, modifierFunction);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
    const std::string& parameterName, const ModifierAction modifierAction)
{
    switch (modifierAction)
    {
    case ModifierAction::Add:
        setThreadModifier(id, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize + parameters.at(0);});
        break;
    case ModifierAction::Subtract:
        setThreadModifier(id, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize - parameters.at(0);});
        break;
    case ModifierAction::Multiply:
        setThreadModifier(id, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize * parameters.at(0);});
        break;
    case ModifierAction::Divide:
        setThreadModifier(id, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize / parameters.at(0);});
        break;
    default:
        throw std::runtime_error("Unknown modifier action");
    }
}

void Tuner::setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    try
    {
        tunerCore->setLocalMemoryModifier(id, argumentId, parameterNames, modifierFunction);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::addConstraint(const KernelId id, const std::vector<std::string>& parameterNames,
    const std::function<bool(const std::vector<size_t>&)>& constraintFunction)
{
    try
    {
        tunerCore->addConstraint(id, parameterNames, constraintFunction);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setTuningManipulatorSynchronization(const KernelId id, const bool flag)
{
    try
    {
        tunerCore->setTuningManipulatorSynchronization(id, flag);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
    const ModifierDimension modifierDimension, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    try
    {
        tunerCore->setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, parameterNames, modifierFunction);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
    const ModifierDimension modifierDimension, const std::string& parameterName, const ModifierAction modifierAction)
{
    switch (modifierAction)
    {
    case ModifierAction::Add:
        setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize + parameters.at(0);});
        break;
    case ModifierAction::Subtract:
        setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize - parameters.at(0);});
        break;
    case ModifierAction::Multiply:
        setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize * parameters.at(0);});
        break;
    case ModifierAction::Divide:
        setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, std::vector<std::string>{parameterName},
            [](const size_t threadSize, const std::vector<size_t>& parameters) {return threadSize / parameters.at(0);});
        break;
    default:
        throw std::runtime_error("Unknown modifier action");
    }
}

void Tuner::setCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const ArgumentId argumentId,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    try
    {
        tunerCore->setCompositionKernelLocalMemoryModifier(compositionId, kernelId, argumentId, parameterNames, modifierFunction);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::persistArgument(const ArgumentId id, const bool flag)
{
    try
    {
        tunerCore->persistArgument(id, flag);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::downloadPersistentArgument(const OutputDescriptor& output) const
{
    try
    {
        tunerCore->downloadPersistentArgument(output);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

std::vector<ComputationResult> Tuner::tuneKernel(const KernelId id)
{
    try
    {
        return tunerCore->tuneKernel(id, nullptr);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

std::vector<ComputationResult> Tuner::tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
{
    try
    {
        return tunerCore->tuneKernel(id, std::move(stopCondition));
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

std::vector<ComputationResult> Tuner::dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations)
{
    try
    {
        return tunerCore->dryTuneKernel(id, filePath, iterations);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

ComputationResult Tuner::tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output)
{
    try
    {
        return tunerCore->tuneKernelByStep(id, output, true);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

ComputationResult Tuner::tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference)
{
    try
    {
        return tunerCore->tuneKernelByStep(id, output, recomputeReference);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

ComputationResult Tuner::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output)
{
    try
    {
        return tunerCore->runKernel(id, configuration, output);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::clearKernelData(const KernelId id, const bool clearConfigurations)
{
    tunerCore->clearKernelData(id, clearConfigurations);
}

void Tuner::setKernelProfiling(const bool flag)
{
    try
    {
        tunerCore->setKernelProfiling(flag);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setCompositionKernelProfiling(const KernelId compositionId, const KernelId kernelId, const bool flag)
{
    try
    {
        tunerCore->setCompositionKernelProfiling(compositionId, kernelId, flag);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setKernelProfilingCounters(const std::vector<std::string>& counterNames)
{
    try
    {
        tunerCore->setKernelProfilingCounters(counterNames);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
{
    try
    {
        tunerCore->setSearchMethod(method, arguments);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setPrintingTimeUnit(const TimeUnit unit)
{
    tunerCore->setPrintingTimeUnit(unit);
}

void Tuner::setInvalidResultPrinting(const bool flag)
{
    tunerCore->setInvalidResultPrinting(flag);
}

void Tuner::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const
{
    try
    {
        tunerCore->printResult(id, outputTarget, format);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
    }
}

void Tuner::printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const
{
    try
    {
        tunerCore->printResult(id, filePath, format);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
    }
}

ComputationResult Tuner::getBestComputationResult(const KernelId id) const
{
    try
    {
        return tunerCore->getBestComputationResult(id);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

std::string Tuner::getKernelSource(const KernelId id, const std::vector<ParameterPair>& configuration) const
{
    try
    {
        return tunerCore->getKernelSource(id, configuration);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
    }
}

void Tuner::setValidationMode(const ValidationMode mode)
{
    tunerCore->setValidationMode(mode);
}

void Tuner::setValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    try
    {
        tunerCore->setValidationMethod(method, toleranceThreshold);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
    }
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

void Tuner::setKernelCacheCapacity(const size_t capacity)
{
    tunerCore->setKernelCacheCapacity(capacity);
}

void Tuner::printComputeAPIInfo(std::ostream& outputTarget) const
{
    try
    {
        tunerCore->printComputeAPIInfo(outputTarget);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

std::vector<DeviceInfo> Tuner::getDeviceInfo(const PlatformIndex platform) const
{
    try
    {
        return tunerCore->getDeviceInfo(platform);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
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
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

void Tuner::setAutomaticGlobalSizeCorrection(const bool flag)
{
    tunerCore->setAutomaticGlobalSizeCorrection(flag);
}

void Tuner::setGlobalSizeType(const GlobalSizeType type)
{
    tunerCore->setGlobalSizeType(type);
}

void Tuner::setLoggingLevel(const LoggingLevel level)
{
    TunerCore::setLoggingLevel(level);
}

void Tuner::setLoggingTarget(std::ostream& outputTarget)
{
    TunerCore::setLoggingTarget(outputTarget);
}

void Tuner::setLoggingTarget(const std::string& filePath)
{
    TunerCore::setLoggingTarget(filePath);
}

ArgumentId Tuner::addArgument(void* vectorData, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const bool copyData)
{
    try
    {
        return tunerCore->addArgument(vectorData, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType,
            ArgumentUploadType::Vector, copyData);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

ArgumentId Tuner::addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType)
{
    try
    {
        return tunerCore->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

ArgumentId Tuner::addArgument(const size_t localMemoryElementsCount, const size_t elementSizeInBytes, const ArgumentDataType dataType)
{
    try
    {
        return tunerCore->addArgument(nullptr, localMemoryElementsCount, elementSizeInBytes, dataType, ArgumentMemoryLocation::Device,
            ArgumentAccessType::ReadOnly, ArgumentUploadType::Local);
    }
    catch (const std::runtime_error& error)
    {
        TunerCore::log(LoggingLevel::Error, error.what());
        throw;
    }
}

} // namespace ktt
