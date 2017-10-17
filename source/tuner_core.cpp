#include "tuner_core.h"
#include "compute_engine/cuda/cuda_core.h"
#include "compute_engine/opencl/opencl_core.h"
#include "compute_engine/vulkan/vulkan_core.h"
#include "utility/ktt_utility.h"

namespace ktt
{

TunerCore::TunerCore(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode) :
    argumentManager(std::make_unique<ArgumentManager>(runMode)),
    kernelManager(std::make_unique<KernelManager>())
{
    if (computeApi == ComputeApi::Opencl)
    {
        computeEngine = std::make_unique<OpenclCore>(platformIndex, deviceIndex, runMode);
    }
    else if (computeApi == ComputeApi::Cuda)
    {
        computeEngine = std::make_unique<CudaCore>(deviceIndex, runMode);
        kernelManager->setGlobalSizeType(GlobalSizeType::Cuda);
        resultPrinter.setGlobalSizeType(GlobalSizeType::Cuda);
    }
    else if (computeApi == ComputeApi::Vulkan)
    {
        computeEngine = std::make_unique<VulkanCore>(deviceIndex);
        kernelManager->setGlobalSizeType(GlobalSizeType::Vulkan);
        resultPrinter.setGlobalSizeType(GlobalSizeType::Vulkan);
    }
    else
    {
        throw std::runtime_error("Specified compute API is not supported");
    }

    tuningRunner = std::make_unique<TuningRunner>(argumentManager.get(), kernelManager.get(), &logger, computeEngine.get(), runMode);

    if (computeApi == ComputeApi::Cuda)
    {
        tuningRunner->setGlobalSizeType(GlobalSizeType::Cuda);
    }
    else if (computeApi == ComputeApi::Vulkan)
    {
        tuningRunner->setGlobalSizeType(GlobalSizeType::Vulkan);
    }

    DeviceInfo info = computeEngine->getCurrentDeviceInfo();
    logger.log(std::string("Initializing tuner for device: ") + info.getName());
}

size_t TunerCore::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernel(source, kernelName, globalSize, localSize);
}

size_t TunerCore::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernelFromFile(filePath, kernelName, globalSize, localSize);
}

size_t TunerCore::addKernelComposition(const std::vector<size_t>& kernelIds, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    size_t compositionId = kernelManager->addKernelComposition(kernelIds);
    tuningRunner->setTuningManipulator(compositionId, std::move(tuningManipulator));
    return compositionId;
}

void TunerCore::addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    kernelManager->addParameter(kernelId, parameterName, parameterValues, threadModifierType, threadModifierAction, modifierDimension);
}

void TunerCore::addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    kernelManager->addConstraint(kernelId, constraintFunction, parameterNames);
}

void TunerCore::setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIndices)
{
    for (const auto index : argumentIndices)
    {
        if (index >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: ") + std::to_string(index));
        }
    }

    if (!containsUnique(argumentIndices))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    kernelManager->setArguments(kernelId, argumentIndices);
}

void TunerCore::addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction,
    const Dimension& modifierDimension)
{
    kernelManager->addCompositionKernelParameter(compositionId, kernelId, parameterName, parameterValues, threadModifierType, threadModifierAction,
        modifierDimension);
}

void TunerCore::setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    for (const auto index : argumentIds)
    {
        if (index >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: ") + std::to_string(index));
        }
    }

    if (!containsUnique(argumentIds))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    kernelManager->setCompositionKernelArguments(compositionId, kernelId, argumentIds);
}

void TunerCore::setGlobalSizeType(const GlobalSizeType& globalSizeType)
{
    kernelManager->setGlobalSizeType(globalSizeType);
    resultPrinter.setGlobalSizeType(globalSizeType);
}

size_t TunerCore::addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType)
{
    return argumentManager->addArgument(data, numberOfElements, dataType, memoryLocation, accessType, uploadType);
}

void TunerCore::tuneKernel(const size_t kernelId)
{
    std::vector<TuningResult> results = tuningRunner->tuneKernel(kernelId);
    resultPrinter.setResult(kernelId, results);
}

void TunerCore::runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    tuningRunner->runKernelPublic(kernelId, kernelConfiguration, outputDescriptors);
}

void TunerCore::setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    tuningRunner->setSearchMethod(searchMethod, searchArguments);
}

void TunerCore::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    tuningRunner->setValidationMethod(validationMethod, toleranceThreshold);
}

void TunerCore::setValidationRange(const size_t argumentId, const size_t validationRange)
{
    if (argumentId > argumentManager->getArgumentCount())
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(argumentId));
    }
    if (validationRange > argumentManager->getArgument(argumentId).getNumberOfElements())
    {
        throw std::runtime_error(std::string("Invalid validation range for argument with id: ") + std::to_string(argumentId));
    }
    tuningRunner->setValidationRange(argumentId, validationRange);
}

void TunerCore::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    if (!kernelManager->isKernel(kernelId) && !kernelManager->isKernelComposition(kernelId))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }
    if (!kernelManager->isKernel(referenceKernelId) || kernelManager->getKernel(referenceKernelId).hasTuningManipulator())
    {
        throw std::runtime_error(std::string("Reference kernel cannot be composite and cannot use tuning manipulator: ") + std::to_string(kernelId));
    }
    tuningRunner->setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
}

void TunerCore::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    if (!kernelManager->isKernel(kernelId) && !kernelManager->isKernelComposition(kernelId))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }
    tuningRunner->setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
}

void TunerCore::setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    if (!kernelManager->isKernel(kernelId) && !kernelManager->isKernelComposition(kernelId))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }
    tuningRunner->setTuningManipulator(kernelId, std::move(tuningManipulator));

    if (kernelManager->isKernel(kernelId))
    {
        kernelManager->getKernel(kernelId).setTuningManipulatorFlag(true);
    }
}

void TunerCore::enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)
{
    if (argumentId >= argumentManager->getArgumentCount())
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(argumentId));
    }
    tuningRunner->enableArgumentPrinting(argumentId, filePath, argumentPrintCondition);
}

void TunerCore::setPrintingTimeUnit(const TimeUnit& timeUnit)
{
    resultPrinter.setTimeUnit(timeUnit);
}

void TunerCore::setInvalidResultPrinting(const bool flag)
{
    resultPrinter.setInvalidResultPrinting(flag);
}

void TunerCore::printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
{
    resultPrinter.printResult(kernelId, outputTarget, printFormat);
}

void TunerCore::printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const
{
    std::ofstream outputFile(filePath);

    if (!outputFile.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    resultPrinter.printResult(kernelId, outputFile, printFormat);
}

std::vector<ParameterValue> TunerCore::getBestConfiguration(const size_t kernelId) const
{
    return resultPrinter.getBestConfiguration(kernelId);
}

void TunerCore::setCompilerOptions(const std::string& options)
{
    computeEngine->setCompilerOptions(options);
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
