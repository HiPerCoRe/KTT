#include <string>
#include <utility>
#include "result_validator.h"
#include "utility/ktt_utility.h"

namespace ktt
{

ResultValidator::ResultValidator(ArgumentManager* argumentManager, KernelRunner* kernelRunner, Logger* logger) :
    argumentManager(argumentManager),
    kernelRunner(kernelRunner),
    logger(logger),
    toleranceThreshold(1e-4),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

void ResultValidator::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (referenceKernels.find(id) != referenceKernels.end())
    {
        referenceKernels.erase(id);
    }
    referenceKernels.insert(std::make_pair(id, std::make_tuple(referenceId, referenceConfiguration, validatedArgumentIds)));
}

void ResultValidator::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (referenceClasses.find(id) != referenceClasses.end())
    {
        referenceClasses.erase(id);
    }
    referenceClasses.insert(std::make_pair(id, std::make_tuple(std::move(referenceClass), validatedArgumentIds)));
}

void ResultValidator::setToleranceThreshold(const double threshold)
{
    if (threshold < 0.0)
    {
        throw std::runtime_error("Tolerance threshold cannot be negative");
    }
    this->toleranceThreshold = threshold;
}

void ResultValidator::setValidationMethod(const ValidationMethod method)
{
    this->validationMethod = method;
}

void ResultValidator::setValidationRange(const ArgumentId id, const size_t range)
{
    if (argumentValidationRanges.find(id) != argumentValidationRanges.end())
    {
        argumentValidationRanges.erase(id);
    }
    argumentValidationRanges.insert(std::make_pair(id, range));
}

void ResultValidator::setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
{
    if (argumentComparators.find(id) != argumentComparators.end())
    {
        argumentComparators.erase(id);
    }
    argumentComparators.insert(std::make_pair(id, comparator));
}

void ResultValidator::computeReferenceResult(const Kernel& kernel)
{
    computeReferenceResultWithClass(kernel);
    computeReferenceResultWithKernel(kernel);
}

void ResultValidator::clearReferenceResults()
{
    referenceClassResults.clear();
    referenceKernelResults.clear();
}

bool ResultValidator::validateArgumentsWithClass(const Kernel& kernel)
{
    KernelId kernelId = kernel.getId();

    auto referenceClassPointer = referenceClasses.find(kernelId);
    if (referenceClassPointer == referenceClasses.end())
    {
        return true; // reference class not present, no validation required
    }

    std::vector<ArgumentId> argumentIds = std::get<1>(referenceClassPointer->second);
    std::vector<KernelArgument> resultArguments;

    for (const auto argumentId : argumentIds)
    {
        KernelArgument resultArgument = kernelRunner->downloadArgument(argumentId);
        resultArguments.push_back(resultArgument);
    }

    return validateArguments(resultArguments, referenceClassResults.find(kernelId)->second, kernel.getName());
}

bool ResultValidator::validateArgumentsWithKernel(const Kernel& kernel)
{
    KernelId kernelId = kernel.getId();

    auto referenceKernelPointer = referenceKernels.find(kernelId);
    if (referenceKernelPointer == referenceKernels.end())
    {
        return true; // reference kernel not present, no validation required
    }

    std::vector<ArgumentId> argumentIds = std::get<2>(referenceKernelPointer->second);
    std::vector<KernelArgument> resultArguments;

    for (const auto argumentId : argumentIds)
    {
        KernelArgument resultArgument = kernelRunner->downloadArgument(argumentId);
        resultArguments.push_back(resultArgument);
    }

    return validateArguments(resultArguments, referenceKernelResults.find(kernelId)->second, kernel.getName());
}

double ResultValidator::getToleranceThreshold() const
{
    return toleranceThreshold;
}

ValidationMethod ResultValidator::getValidationMethod() const
{
    return validationMethod;
}

void ResultValidator::computeReferenceResultWithClass(const Kernel& kernel)
{
    KernelId kernelId = kernel.getId();

    auto referenceClassPointer = referenceClasses.find(kernelId);
    if (referenceClassPointer == referenceClasses.end())
    {
        return; // reference class not present
    }

    ReferenceClass* referenceClass = std::get<0>(referenceClassPointer->second).get();
    std::vector<ArgumentId> referenceArgumentIds = std::get<1>(referenceClassPointer->second);

    for (const auto argumentId : referenceArgumentIds)
    {
        if (!elementExists(argumentId, kernel.getArgumentIds()))
        {
            throw std::runtime_error(std::string("Reference argument with id: ") + std::to_string(argumentId)
                + " is not assciated with kernel with id: " + std::to_string(kernel.getId()));
        }
        if (argumentManager->getArgument(argumentId).getAccessType() == ArgumentAccessType::ReadOnly)
        {
            throw std::runtime_error(std::string("Reference argument with following id is marked as read only: ") + std::to_string(argumentId));
        }
    }

    logger->log(std::string("Computing reference class result for kernel: ") + kernel.getName());
    referenceClass->computeResult();
    std::vector<KernelArgument> referenceResult;

    for (const auto referenceArgumentId : referenceArgumentIds)
    {
        size_t numberOfElements = referenceClass->getNumberOfElements(referenceArgumentId);
        const auto& argument = argumentManager->getArgument(referenceArgumentId);

        if (numberOfElements == 0)
        {
            numberOfElements = argument.getNumberOfElements();
        }

        referenceResult.emplace_back(referenceArgumentId, referenceClass->getData(referenceArgumentId), numberOfElements,
            argument.getElementSizeInBytes(), argument.getDataType(), argument.getMemoryLocation(), argument.getAccessType(),
            argument.getUploadType(), false);
    }
    referenceClassResults.insert(std::make_pair(kernelId, referenceResult));
}

void ResultValidator::computeReferenceResultWithKernel(const Kernel& kernel)
{
    KernelId kernelId = kernel.getId();

    auto referenceKernelPointer = referenceKernels.find(kernelId);
    if (referenceKernelPointer == referenceKernels.end())
    {
        return; // reference kernel not present
    }

    KernelId referenceKernelId = std::get<0>(referenceKernelPointer->second);
    std::vector<ParameterPair> referenceParameters = std::get<1>(referenceKernelPointer->second);
    std::vector<ArgumentId> referenceArgumentIds = std::get<2>(referenceKernelPointer->second);

    for (const auto argumentId : referenceArgumentIds)
    {
        if (!elementExists(argumentId, kernel.getArgumentIds()))
        {
            throw std::runtime_error(std::string("Reference argument with id: ") + std::to_string(argumentId)
                + " is not assciated with kernel with id: " + std::to_string(kernel.getId()));
        }
        if (argumentManager->getArgument(argumentId).getAccessType() == ArgumentAccessType::ReadOnly)
        {
            throw std::runtime_error(std::string("Reference argument with following id is marked as read only: ") + std::to_string(argumentId));
        }
    }

    logger->log(std::string("Computing reference kernel result for kernel: ") + kernel.getName());
    kernelRunner->runKernel(referenceKernelId, referenceParameters, std::vector<OutputDescriptor>{});

    std::vector<KernelArgument> referenceResult;
    for (const auto argumentId : referenceArgumentIds)
    {
        referenceResult.push_back(kernelRunner->downloadArgument(argumentId));
    }

    kernelRunner->clearBuffers();
    referenceKernelResults.insert(std::make_pair(kernelId, referenceResult));
}

bool ResultValidator::validateArguments(const std::vector<KernelArgument>& resultArguments, const std::vector<KernelArgument>& referenceArguments,
    const std::string kernelName) const
{
    bool validationResult = true;

    for (const auto& referenceArgument : referenceArguments)
    {
        bool argumentValidated = false;

        for (const auto& resultArgument : resultArguments)
        {
            if (resultArgument.getId() != referenceArgument.getId())
            {
                continue;
            }

            ArgumentDataType referenceDataType = referenceArgument.getDataType();
            ArgumentId id = resultArgument.getId();
            if (referenceDataType != resultArgument.getDataType())
            {
                logger->log(std::string("Reference class argument data type mismatch for argument id: ") + std::to_string(resultArgument.getId()));
                return false;
            }

            bool currentResult;
            auto comparatorPointer = argumentComparators.find(id);

            if (comparatorPointer != argumentComparators.end())
            {
                size_t resultSize = resultArgument.getNumberOfElements();
                size_t referenceSize = referenceArgument.getNumberOfElements();
                auto argumentRangePointer = argumentValidationRanges.find(id);
                if (argumentRangePointer == argumentValidationRanges.end() && resultSize != referenceSize)
                {
                    logger->log(std::string("Number of elements in results differs for argument with id: ") + std::to_string(id)
                        + ", reference size: " + std::to_string(referenceSize) + ", result size: " + std::to_string(resultSize));
                    currentResult = false;
                }
                else if (argumentRangePointer != argumentValidationRanges.end())
                {
                    currentResult = validateResultCustom(id, resultArgument.getData(), referenceArgument.getData(), argumentRangePointer->second,
                        resultArgument.getElementSizeInBytes(), comparatorPointer->second);
                }
                else
                {
                    currentResult = validateResultCustom(id, resultArgument.getData(), referenceArgument.getData(), resultSize,
                        resultArgument.getElementSizeInBytes(), comparatorPointer->second);
                }
            }
            else if (referenceDataType == ArgumentDataType::Char)
            {
                currentResult = validateResult(resultArgument.getDataWithType<int8_t>(), referenceArgument.getDataWithType<int8_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::UnsignedChar)
            {
                currentResult = validateResult(resultArgument.getDataWithType<uint8_t>(), referenceArgument.getDataWithType<uint8_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Short)
            {
                currentResult = validateResult(resultArgument.getDataWithType<int16_t>(), referenceArgument.getDataWithType<int16_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::UnsignedShort)
            {
                currentResult = validateResult(resultArgument.getDataWithType<uint16_t>(), referenceArgument.getDataWithType<uint16_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Int)
            {
                currentResult = validateResult(resultArgument.getDataWithType<int32_t>(), referenceArgument.getDataWithType<int32_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::UnsignedInt)
            {
                currentResult = validateResult(resultArgument.getDataWithType<uint32_t>(), referenceArgument.getDataWithType<uint32_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Long)
            {
                currentResult = validateResult(resultArgument.getDataWithType<int64_t>(), referenceArgument.getDataWithType<int64_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::UnsignedLong)
            {
                currentResult = validateResult(resultArgument.getDataWithType<uint64_t>(), referenceArgument.getDataWithType<uint64_t>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Half)
            {
                currentResult = validateResult(resultArgument.getDataWithType<half>(), referenceArgument.getDataWithType<half>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Float)
            {
                currentResult = validateResult(resultArgument.getDataWithType<float>(), referenceArgument.getDataWithType<float>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Double)
            {
                currentResult = validateResult(resultArgument.getDataWithType<double>(), referenceArgument.getDataWithType<double>(), id);
            }
            else if (referenceDataType == ArgumentDataType::Custom)
            {
                throw std::runtime_error("Validation of custom data type arguments requires usage of argument comparator");
            }
            else
            {
                throw std::runtime_error("Unsupported argument data type");
            }

            validationResult &= currentResult;
            argumentValidated = true;
        }

        if (!argumentValidated)
        {
            logger->log(std::string("Result for validated argument with following id not found: ") + std::to_string(referenceArgument.getId()));
            return false;
        }
    }

    return validationResult;
}

bool ResultValidator::validateResultCustom(const ArgumentId id, const void* result, const void* referenceResult, const size_t numberOfElements,
    const size_t elementSizeInBytes, const std::function<bool(const void*, const void*)>& comparator) const
{
    for (size_t i = 0; i < numberOfElements * elementSizeInBytes; i += elementSizeInBytes)
    {
        if (!comparator((uint8_t*)result + i, (uint8_t*)referenceResult + i))
        {
            logger->log(std::string("Results differ for argument with id: ") + std::to_string(id) + ", index: "
                + std::to_string(i / elementSizeInBytes));
            return false;
        }
    }

    return true;
}

} // namespace ktt
