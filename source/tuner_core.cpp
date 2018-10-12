#include "tuner_core.h"
#include "compute_engine/cuda/cuda_engine.h"
#include "compute_engine/opencl/opencl_engine.h"
#include "compute_engine/vulkan/vulkan_engine.h"
#include "utility/ktt_utility.h"

namespace ktt
{

TunerCore::TunerCore(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI, const uint32_t queueCount) :
    argumentManager(std::make_unique<ArgumentManager>())
{
    if (queueCount == 0)
    {
        throw std::runtime_error("Number of compute queues must be greater than zero");
    }

    if (computeAPI == ComputeAPI::OpenCL)
    {
        computeEngine = std::make_unique<OpenCLEngine>(platform, device, queueCount);
    }
    else if (computeAPI == ComputeAPI::CUDA)
    {
        computeEngine = std::make_unique<CUDAEngine>(device, queueCount);
    }
    else if (computeAPI == ComputeAPI::Vulkan)
    {
        computeEngine = std::make_unique<VulkanEngine>(device, queueCount);
    }
    else
    {
        throw std::runtime_error("Specified compute API is not supported");
    }

    DeviceInfo info = computeEngine->getCurrentDeviceInfo();
    Logger::getLogger().log(LoggingLevel::Info, std::string("Initializing tuner for device ") + info.getName());

    kernelManager = std::make_unique<KernelManager>();
    kernelRunner = std::make_unique<KernelRunner>(argumentManager.get(), kernelManager.get(), computeEngine.get());
    tuningRunner = std::make_unique<TuningRunner>(argumentManager.get(), kernelManager.get(), kernelRunner.get(), info);
}

KernelId TunerCore::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernel(source, kernelName, globalSize, localSize);
}

KernelId TunerCore::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernelFromFile(filePath, kernelName, globalSize, localSize);
}

KernelId TunerCore::addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
    std::unique_ptr<TuningManipulator> manipulator)
{
    KernelId compositionId = kernelManager->addKernelComposition(compositionName, kernelIds);
    kernelRunner->setTuningManipulator(compositionId, std::move(manipulator));
    return compositionId;
}

void TunerCore::addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
{
    kernelManager->addParameter(id, parameterName, parameterValues);
}

void TunerCore::addParameter(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues)
{
    kernelManager->addParameter(id, parameterName, parameterValues);
}

void TunerCore::addConstraint(const KernelId id, const std::vector<std::string>& parameterNames,
    const std::function<bool(const std::vector<size_t>&)>& constraintFunction)
{
    kernelManager->addConstraint(id, parameterNames, constraintFunction);
}

void TunerCore::addParameterPack(const KernelId id, const std::string& packName, const std::vector<std::string>& parameterNames)
{
    kernelManager->addParameterPack(id, packName, parameterNames);
}

void TunerCore::setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    kernelManager->setThreadModifier(id, modifierType, modifierDimension, parameterNames, modifierFunction);
}

void TunerCore::setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    kernelManager->setLocalMemoryModifier(id, argumentId, parameterNames, modifierFunction);
}

void TunerCore::setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
    const ModifierDimension modifierDimension, const std::vector<std::string>& parameterNames,
    const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    kernelManager->setCompositionKernelThreadModifier(compositionId, kernelId, modifierType, modifierDimension, parameterNames, modifierFunction);
}

void TunerCore::setCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const ArgumentId argumentId,
    const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
{
    kernelManager->setCompositionKernelLocalMemoryModifier(compositionId, kernelId, argumentId, parameterNames, modifierFunction);
}

void TunerCore::setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    for (const auto id : argumentIds)
    {
        if (id >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: ") + std::to_string(id));
        }
    }

    if (!containsUnique(argumentIds))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    kernelManager->setArguments(id, argumentIds);
}

void TunerCore::setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
{
    for (const auto id : argumentIds)
    {
        if (id >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: ") + std::to_string(id));
        }
    }

    if (!containsUnique(argumentIds))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    kernelManager->setCompositionKernelArguments(compositionId, kernelId, argumentIds);
}

std::string TunerCore::getKernelSource(const KernelId id, const std::vector<ParameterPair>& configuration) const
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    return kernelManager->getKernelSourceWithDefines(id, configuration);
}

ArgumentId TunerCore::addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType, const bool copyData)
{
    return argumentManager->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType, copyData);
}

ArgumentId TunerCore::addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType)
{
    return argumentManager->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType);
}

ComputationResult TunerCore::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration,
    const std::vector<OutputDescriptor>& output)
{
    KernelResult result;

    if (kernelManager->isComposition(id))
    {
        result = kernelRunner->runComposition(id, configuration, output);
    }
    else
    {
        result = kernelRunner->runKernel(id, configuration, output);
    }

    kernelRunner->clearBuffers();

    if (result.isValid())
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getComputationDuration());
    }
    else
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getErrorMessage());
    }
}

void TunerCore::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (!kernelManager->isKernel(id) && !kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    kernelRunner->setTuningManipulator(id, std::move(manipulator));

    if (kernelManager->isKernel(id))
    {
        kernelManager->getKernel(id).setTuningManipulatorFlag(true);
    }
}

void TunerCore::setTuningManipulatorSynchronization(const KernelId id, const bool flag)
{
    if (!kernelManager->isKernel(id) && !kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    kernelRunner->setTuningManipulatorSynchronization(id, flag);
}

void TunerCore::tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
{
    std::vector<KernelResult> results;
    if (kernelManager->isComposition(id))
    {
        results = tuningRunner->tuneComposition(id, std::move(stopCondition));
    }
    else
    {
        results = tuningRunner->tuneKernel(id, std::move(stopCondition));
    }
    resultPrinter.setResult(id, results);
}

void TunerCore::dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations)
{
    std::vector<KernelResult> results;
    if (kernelManager->isComposition(id))
    {
        throw std::runtime_error("Dry run is not implemented for compositions");
    }
    else
    {
        results = tuningRunner->dryTuneKernel(id, filePath, iterations);
    }
    resultPrinter.setResult(id, results);
}

ComputationResult TunerCore::tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference)
{
    KernelResult result;

    if (kernelManager->isComposition(id))
    {
        result = tuningRunner->tuneCompositionByStep(id, output, recomputeReference);
    }
    else
    {
        result = tuningRunner->tuneKernelByStep(id, output, recomputeReference);
    }

    resultPrinter.addResult(id, result);
    
    if (result.isValid())
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getComputationDuration());
    }
    else
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getErrorMessage());
    }
}

void TunerCore::clearKernelData(const KernelId id, const bool clearConfigurations)
{
    resultPrinter.clearResults(id);
    tuningRunner->clearKernelData(id, clearConfigurations);
}

void TunerCore::setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
{
    tuningRunner->setSearchMethod(method, arguments);
}

void TunerCore::setValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    tuningRunner->setValidationMethod(method, toleranceThreshold);
}

void TunerCore::setValidationRange(const ArgumentId id, const size_t range)
{
    if (id > argumentManager->getArgumentCount())
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    if (range > argumentManager->getArgument(id).getNumberOfElements())
    {
        throw std::runtime_error(std::string("Invalid validation range for argument with id: ") + std::to_string(id));
    }
    tuningRunner->setValidationRange(id, range);
}

void TunerCore::setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
{
    if (id > argumentManager->getArgumentCount())
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    tuningRunner->setArgumentComparator(id, comparator);
}

void TunerCore::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (!kernelManager->isKernel(id) && !kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    if (!kernelManager->isKernel(referenceId))
    {
        throw std::runtime_error(std::string("Invalid reference kernel id: ") + std::to_string(referenceId));
    }
    tuningRunner->setReferenceKernel(id, referenceId, referenceConfiguration, validatedArgumentIds);
}

void TunerCore::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (!kernelManager->isKernel(id) && !kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    tuningRunner->setReferenceClass(id, std::move(referenceClass), validatedArgumentIds);
}

ComputationResult TunerCore::getBestComputationResult(const KernelId id) const
{
    return tuningRunner->getBestComputationResult(id);
}

void TunerCore::setPrintingTimeUnit(const TimeUnit unit)
{
    resultPrinter.setTimeUnit(unit);
}

void TunerCore::setInvalidResultPrinting(const bool flag)
{
    resultPrinter.setInvalidResultPrinting(flag);
}

void TunerCore::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const
{
    resultPrinter.printResult(id, outputTarget, format);
}

void TunerCore::printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const
{
    std::ofstream outputFile(filePath);

    if (!outputFile.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    resultPrinter.printResult(id, outputFile, format);
}

void TunerCore::setCompilerOptions(const std::string& options)
{
    computeEngine->setCompilerOptions(options);
}

void TunerCore::setGlobalSizeType(const GlobalSizeType type)
{
    computeEngine->setGlobalSizeType(type);
}

void TunerCore::setAutomaticGlobalSizeCorrection(const bool flag)
{
    computeEngine->setAutomaticGlobalSizeCorrection(flag);
}

void TunerCore::setKernelCacheCapacity(const size_t capacity)
{
    if (capacity == 0)
    {
        computeEngine->setKernelCacheUsage(false);
    }
    else
    {
        computeEngine->setKernelCacheUsage(true);
    }
    computeEngine->setKernelCacheCapacity(capacity);
}

void TunerCore::persistArgument(const ArgumentId id, const bool flag)
{
    argumentManager->setPersistentFlag(id, flag);
    KernelArgument& argument = argumentManager->getArgument(id);
    computeEngine->persistArgument(argument, flag);
}

void TunerCore::downloadPersistentArgument(const OutputDescriptor& output) const
{
    computeEngine->downloadArgument(output.getArgumentId(), output.getOutputDestination(), output.getOutputSizeInBytes());
}

void TunerCore::printComputeAPIInfo(std::ostream& outputTarget) const
{
    computeEngine->printComputeAPIInfo(outputTarget);
}

std::vector<PlatformInfo> TunerCore::getPlatformInfo() const
{
    return computeEngine->getPlatformInfo();
}

std::vector<DeviceInfo> TunerCore::getDeviceInfo(const PlatformIndex platform) const
{
    return computeEngine->getDeviceInfo(platform);
}

DeviceInfo TunerCore::getCurrentDeviceInfo() const
{
    return computeEngine->getCurrentDeviceInfo();
}

void TunerCore::setLoggingLevel(const LoggingLevel level)
{
    Logger::getLogger().setLoggingLevel(level);
}

void TunerCore::setLoggingTarget(std::ostream& outputTarget)
{
    Logger::getLogger().setLoggingTarget(outputTarget);
}

void TunerCore::setLoggingTarget(const std::string& filePath)
{
    Logger::getLogger().setLoggingTarget(filePath);
}

void TunerCore::log(const LoggingLevel level, const std::string& message)
{
    Logger::getLogger().log(level, message);
}

} // namespace ktt
