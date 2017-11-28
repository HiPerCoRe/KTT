#include "tuner_core.h"
#include "compute_engine/cuda/cuda_core.h"
#include "compute_engine/opencl/opencl_core.h"
#include "compute_engine/vulkan/vulkan_core.h"
#include "utility/ktt_utility.h"

namespace ktt
{

TunerCore::TunerCore(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi) :
    argumentManager(std::make_unique<ArgumentManager>()),
    kernelManager(std::make_unique<KernelManager>())
{
    if (computeApi == ComputeApi::Opencl)
    {
        computeEngine = std::make_unique<OpenclCore>(platformIndex, deviceIndex);
    }
    else if (computeApi == ComputeApi::Cuda)
    {
        computeEngine = std::make_unique<CudaCore>(deviceIndex);
    }
    else if (computeApi == ComputeApi::Vulkan)
    {
        computeEngine = std::make_unique<VulkanCore>(deviceIndex);
    }
    else
    {
        throw std::runtime_error("Specified compute API is not supported");
    }
    tuningRunner = std::make_unique<TuningRunner>(argumentManager.get(), kernelManager.get(), &logger, computeEngine.get());

    DeviceInfo info = computeEngine->getCurrentDeviceInfo();
    logger.log(std::string("Initializing tuner for device: ") + info.getName());
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
    tuningRunner->setTuningManipulator(compositionId, std::move(manipulator));
    return compositionId;
}

void TunerCore::addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
    const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension)
{
    kernelManager->addParameter(id, parameterName, parameterValues, modifierType, modifierAction, modifierDimension);
}

void TunerCore::addParameter(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues)
{
    kernelManager->addParameter(id, parameterName, parameterValues);
}

void TunerCore::addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    kernelManager->addConstraint(id, constraintFunction, parameterNames);
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

void TunerCore::addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
    const Dimension& modifierDimension)
{
    kernelManager->addCompositionKernelParameter(compositionId, kernelId, parameterName, parameterValues, modifierType, modifierAction,
        modifierDimension);
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

ArgumentId TunerCore::addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType, const bool copyData)
{
    return argumentManager->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType, copyData);
}

ArgumentId TunerCore::addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
    const ArgumentDataType& dataType, const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType,
    const ArgumentUploadType& uploadType)
{
    return argumentManager->addArgument(data, numberOfElements, elementSizeInBytes, dataType, memoryLocation, accessType, uploadType);
}

void TunerCore::tuneKernel(const KernelId id)
{
    std::vector<TuningResult> results;
    if (kernelManager->isComposition(id))
    {
        results = tuningRunner->tuneComposition(id);
    }
    else
    {
        results = tuningRunner->tuneKernel(id);
    }
    resultPrinter.setResult(id, results);
}

void TunerCore::dryTuneKernel(const KernelId id, const std::string& filePath)
{
    std::vector<TuningResult> results;
    if (kernelManager->isComposition(id))
    {
        throw std::runtime_error("Dry run is not implemented for compositions");
    }
    else
    {
        results = tuningRunner->dryTuneKernel(id, filePath);
    }
    resultPrinter.setResult(id, results);
}

void TunerCore::tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
{
    TuningResult result;
    if (kernelManager->isComposition(id))
    {
        result = tuningRunner->tuneCompositionByStep(id, output);
    }
    else
    {
        result = tuningRunner->tuneKernelByStep(id, output);
    }
    resultPrinter.addResult(id, result);
}

void TunerCore::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (kernelManager->isComposition(id))
    {
        tuningRunner->runComposition(id, configuration, output);
    }
    else
    {
        tuningRunner->runKernel(id, configuration, output);
    }
}

void TunerCore::setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments)
{
    tuningRunner->setSearchMethod(method, arguments);
}

void TunerCore::setValidationMethod(const ValidationMethod& method, const double toleranceThreshold)
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
    if (!kernelManager->isKernel(referenceId) || kernelManager->getKernel(referenceId).hasTuningManipulator())
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

void TunerCore::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (!kernelManager->isKernel(id) && !kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    tuningRunner->setTuningManipulator(id, std::move(manipulator));

    if (kernelManager->isKernel(id))
    {
        kernelManager->getKernel(id).setTuningManipulatorFlag(true);
    }
}

std::vector<ParameterPair> TunerCore::getBestConfiguration(const KernelId id) const
{
    return tuningRunner->getBestConfiguration(id);
}

void TunerCore::setPrintingTimeUnit(const TimeUnit& unit)
{
    resultPrinter.setTimeUnit(unit);
}

void TunerCore::setInvalidResultPrinting(const bool flag)
{
    resultPrinter.setInvalidResultPrinting(flag);
}

void TunerCore::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const
{
    resultPrinter.printResult(id, outputTarget, format);
}

void TunerCore::printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const
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

void TunerCore::setGlobalSizeType(const GlobalSizeType& type)
{
    computeEngine->setGlobalSizeType(type);
}

void TunerCore::setAutomaticGlobalSizeCorrection(const bool flag)
{
    computeEngine->setAutomaticGlobalSizeCorrection(flag);
}

void TunerCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    computeEngine->printComputeApiInfo(outputTarget);
}

std::vector<PlatformInfo> TunerCore::getPlatformInfo() const
{
    return computeEngine->getPlatformInfo();
}

std::vector<DeviceInfo> TunerCore::getDeviceInfo(const size_t platformIndex) const
{
    return computeEngine->getDeviceInfo(platformIndex);
}

DeviceInfo TunerCore::getCurrentDeviceInfo() const
{
    return computeEngine->getCurrentDeviceInfo();
}

void TunerCore::setLoggingTarget(std::ostream& outputTarget)
{
    logger.setLoggingTarget(outputTarget);
}

void TunerCore::setLoggingTarget(const std::string& filePath)
{
    logger.setLoggingTarget(filePath);
}

void TunerCore::log(const std::string& message) const
{
    logger.log(message);
}

} // namespace ktt
