#include "tuner_core.h"
#include "compute_api_driver/opencl/opencl_core.h"

namespace ktt
{

TunerCore::TunerCore(const size_t platformIndex, const size_t deviceIndex) :
    argumentManager(std::make_unique<ArgumentManager>()),
    kernelManager(std::make_unique<KernelManager>()),
    computeApiDriver(std::make_unique<OpenclCore>(platformIndex, deviceIndex)),
    tuningRunner(std::make_unique<TuningRunner>(argumentManager.get(), kernelManager.get(), &logger, computeApiDriver.get()))
{}

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

void TunerCore::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    kernelManager->addParameter(id, name, values, threadModifierType, threadModifierAction, modifierDimension);
}

void TunerCore::addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    kernelManager->addConstraint(id, constraintFunction, parameterNames);
}

void TunerCore::setKernelArguments(const size_t id, const std::vector<size_t>& argumentIndices)
{
    for (const auto index : argumentIndices)
    {
        if (index >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: ") + std::to_string(index));
        }
    }

    kernelManager->setArguments(id, argumentIndices);
}

void TunerCore::setSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    kernelManager->setSearchMethod(id, searchMethod, searchArguments);
}

void TunerCore::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    kernelManager->setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
}

void TunerCore::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    kernelManager->setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
}

void TunerCore::setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    kernelManager->setTuningManipulator(kernelId, std::move(tuningManipulator));
}

void TunerCore::tuneKernel(const size_t id)
{
    auto result = tuningRunner->tuneKernel(id);
    resultPrinter.setResult(id, result);
}

void TunerCore::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    tuningRunner->setValidationMethod(validationMethod, toleranceThreshold);
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

void TunerCore::setCompilerOptions(const std::string& options)
{
    computeApiDriver->setCompilerOptions(options);
}

void TunerCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    computeApiDriver->printComputeApiInfo(outputTarget);
}

std::vector<PlatformInfo> TunerCore::getPlatformInfo() const
{
    return computeApiDriver->getPlatformInfo();
}

std::vector<DeviceInfo> TunerCore::getDeviceInfo(const size_t platformIndex) const
{
    return computeApiDriver->getDeviceInfo(platformIndex);
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
