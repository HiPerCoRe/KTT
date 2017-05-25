#include <string>
#include <utility>

#include "../utility/ktt_utility.h"
#include "result_validator.h"

namespace ktt
{

ResultValidator::ResultValidator(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger,
    ComputeApiDriver* computeApiDriver) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeApiDriver(computeApiDriver),
    argumentPrinter(logger),
    toleranceThreshold(1e-4),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

void ResultValidator::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    if (referenceKernelMap.find(kernelId) != referenceKernelMap.end())
    {
        referenceKernelMap.erase(kernelId);
    }
    referenceKernelMap.insert(std::make_pair(kernelId, std::make_tuple(referenceKernelId, referenceKernelConfiguration, resultArgumentIds)));
}

void ResultValidator::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    if (referenceClassMap.find(kernelId) != referenceClassMap.end())
    {
        referenceClassMap.erase(kernelId);
    }
    referenceClassMap.insert(std::make_pair(kernelId, std::make_tuple(std::move(referenceClass), resultArgumentIds)));
}

void ResultValidator::setToleranceThreshold(const double toleranceThreshold)
{
    if (toleranceThreshold < 0.0)
    {
        throw std::runtime_error("Tolerance threshold cannot be negative");
    }
    this->toleranceThreshold = toleranceThreshold;
}

void ResultValidator::setValidationMethod(const ValidationMethod& validationMethod)
{
    this->validationMethod = validationMethod;
}

void ResultValidator::setValidationRange(const size_t argumentId, const size_t validationRange)
{
    if (argumentValidationRangeMap.find(argumentId) != argumentValidationRangeMap.end())
    {
        argumentValidationRangeMap.erase(argumentId);
    }
    argumentValidationRangeMap.insert(std::make_pair(argumentId, validationRange));
}

void ResultValidator::enableArgumentPrinting(const size_t argumentId, const std::string& filePath,
    const ArgumentPrintCondition& argumentPrintCondition)
{
    argumentPrinter.setArgumentPrintData(argumentId, filePath, argumentPrintCondition);
}

bool ResultValidator::validateArgumentWithClass(const Kernel* kernel, const std::vector<KernelArgument>& resultArguments,
    const KernelConfiguration& kernelConfiguration)
{
    size_t kernelId = kernel->getId();
    if (referenceClassResultMap.find(kernelId) == referenceClassResultMap.end())
    {
        auto referenceClassPointer = referenceClassMap.find(kernelId);
        if (referenceClassPointer == referenceClassMap.end())
        {
            return true; // reference class not present, no validation required
        }

        ReferenceClass* referenceClass = std::get<0>(referenceClassPointer->second).get();
        std::vector<size_t> referenceArgumentIndices = std::get<1>(referenceClassPointer->second);
        for (const auto argumentId : referenceArgumentIndices)
        {
            if (!elementExists(argumentId, kernel->getArgumentIndices()))
            {
                throw std::runtime_error(std::string("Reference argument with id: ") + std::to_string(argumentId)
                    + " is not assciated with kernel with id: " + std::to_string(kernel->getId()));
            }
            if (argumentManager->getArgument(argumentId).getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
            {
                throw std::runtime_error(std::string("Reference argument with following id is marked as read only: ") + std::to_string(argumentId));
            }
        }

        logger->log(std::string("Computing reference class result for kernel: ") + kernel->getName());
        referenceClass->computeResult();
        std::vector<KernelArgument> referenceResult;

        for (const auto referenceArgumentId : referenceArgumentIndices)
        {
            const auto& argument = argumentManager->getArgument(referenceArgumentId);
            referenceResult.emplace_back(KernelArgument(referenceArgumentId, referenceClass->getData(referenceArgumentId),
                referenceClass->getNumberOfElements(referenceArgumentId), argument.getArgumentDataType(), argument.getArgumentMemoryType(),
                argument.getArgumentUploadType()));
        }
        referenceClassResultMap.insert(std::make_pair(kernelId, referenceResult));
    }

    return validateArguments(resultArguments, referenceClassResultMap.find(kernelId)->second, kernel->getName(), kernelConfiguration);
}

bool ResultValidator::validateArgumentWithKernel(const Kernel* kernel, const std::vector<KernelArgument>& resultArguments,
    const KernelConfiguration& kernelConfiguration)
{
    size_t kernelId = kernel->getId();
    if (referenceKernelResultMap.find(kernelId) == referenceKernelResultMap.end())
    {
        auto referenceKernelPointer = referenceKernelMap.find(kernelId);
        if (referenceKernelPointer == referenceKernelMap.end())
        {
            return true; // reference kernel not present, no validation required
        }

        size_t referenceKernelId = std::get<0>(referenceKernelPointer->second);
        std::vector<ParameterValue> referenceParameters = std::get<1>(referenceKernelPointer->second);
        std::vector<size_t> referenceArgumentIndices = std::get<2>(referenceKernelPointer->second);
        for (const auto argumentId : referenceArgumentIndices)
        {
            if (!elementExists(argumentId, kernel->getArgumentIndices()))
            {
                throw std::runtime_error(std::string("Reference argument with id: ") + std::to_string(argumentId)
                    + " is not assciated with kernel with id: " + std::to_string(kernel->getId()));
            }
            if (argumentManager->getArgument(argumentId).getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
            {
                throw std::runtime_error(std::string("Reference argument with following id is marked as read only: ") + std::to_string(argumentId));
            }
        }

        const Kernel* referenceKernel = kernelManager->getKernel(referenceKernelId);
        KernelConfiguration configuration = kernelManager->getKernelConfiguration(referenceKernelId, referenceParameters);
        std::string source = kernelManager->getKernelSourceWithDefines(referenceKernelId, configuration);

        logger->log(std::string("Computing reference kernel result for kernel: ") + kernel->getName());
        auto result = computeApiDriver->runKernel(source, referenceKernel->getName(), convertDimensionVector(configuration.getGlobalSize()),
            convertDimensionVector(configuration.getLocalSize()), getKernelArgumentPointers(referenceKernelId));
        std::vector<KernelArgument> referenceResult;

        for (const auto& argument : result.getResultArguments())
        {
            if (elementExists(argument.getId(), referenceArgumentIndices))
            {
                referenceResult.push_back(argument);
            }
        }
        referenceKernelResultMap.insert(std::make_pair(kernelId, referenceResult));
    }

    return validateArguments(resultArguments, referenceKernelResultMap.find(kernelId)->second, kernel->getName(), kernelConfiguration);
}

void ResultValidator::clearReferenceResults()
{
    referenceClassResultMap.clear();
    referenceKernelResultMap.clear();
}

double ResultValidator::getToleranceThreshold() const
{
    return toleranceThreshold;
}

ValidationMethod ResultValidator::getValidationMethod() const
{
    return validationMethod;
}

bool ResultValidator::validateArguments(const std::vector<KernelArgument>& resultArguments,
    const std::vector<KernelArgument>& referenceArguments, const std::string kernelName, const KernelConfiguration& kernelConfiguration) const
{
    bool validationResult = true;

    for (const auto& referenceArgument : referenceArguments)
    {
        for (const auto& resultArgument : resultArguments)
        {
            if (resultArgument.getId() != referenceArgument.getId())
            {
                continue;
            }

            ArgumentDataType referenceDataType = referenceArgument.getArgumentDataType();

            if (referenceDataType != resultArgument.getArgumentDataType())
            {
                throw std::runtime_error(std::string("Reference class argument data type mismatch for argument id: ")
                    + std::to_string(resultArgument.getId()));
            }

            bool currentResult = true;
            if (referenceDataType == ArgumentDataType::Char)
            {
                currentResult &= validateResult(resultArgument.getDataChar(), referenceArgument.getDataChar(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::UnsignedChar)
            {
                currentResult &= validateResult(resultArgument.getDataUnsignedChar(), referenceArgument.getDataUnsignedChar(),
                    resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Short)
            {
                currentResult &= validateResult(resultArgument.getDataShort(), referenceArgument.getDataShort(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::UnsignedShort)
            {
                currentResult &= validateResult(resultArgument.getDataUnsignedShort(), referenceArgument.getDataUnsignedShort(),
                    resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Int)
            {
                currentResult &= validateResult(resultArgument.getDataInt(), referenceArgument.getDataInt(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::UnsignedInt)
            {
                currentResult &= validateResult(resultArgument.getDataUnsignedInt(), referenceArgument.getDataUnsignedInt(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Long)
            {
                currentResult &= validateResult(resultArgument.getDataLong(), referenceArgument.getDataLong(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::UnsignedLong)
            {
                currentResult &= validateResult(resultArgument.getDataUnsignedLong(), referenceArgument.getDataUnsignedLong(),
                    resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Half)
            {
                currentResult &= validateResult(resultArgument.getDataHalf(), referenceArgument.getDataHalf(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Float)
            {
                currentResult &= validateResult(resultArgument.getDataFloat(), referenceArgument.getDataFloat(), resultArgument.getId());
            }
            else if (referenceDataType == ArgumentDataType::Double)
            {
                currentResult &= validateResult(resultArgument.getDataDouble(), referenceArgument.getDataDouble(), resultArgument.getId());
            }
            else
            {
                throw std::runtime_error("Unsupported argument data type");
            }

            if (argumentPrinter.argumentPrintDataExists(resultArgument.getId()))
            {
                argumentPrinter.printArgument(resultArgument, kernelName, kernelConfiguration, currentResult);
            }
            validationResult &= currentResult;
        }
    }

    return validationResult;
}

std::vector<const KernelArgument*> ResultValidator::getKernelArgumentPointers(const size_t kernelId) const
{
    std::vector<const KernelArgument*> result;

    std::vector<size_t> argumentIndices = kernelManager->getKernel(kernelId)->getArgumentIndices();
    
    for (const auto index : argumentIndices)
    {
        result.push_back(&argumentManager->getArgument(index));
    }

    return result;
}

} // namespace ktt
