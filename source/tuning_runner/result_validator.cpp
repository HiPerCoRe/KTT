#include <string>

#include "result_validator.h"

namespace ktt
{

ResultValidator::ResultValidator() :
    toleranceThreshold(1e-4),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

bool ResultValidator::validateArgumentWithClass(const size_t kernelId, const std::vector<KernelArgument>& resultArguments) const
{
    if (referenceClassResultMap.find(kernelId) == referenceClassResultMap.end())
    {
        throw std::runtime_error(std::string("No reference class results found for kernel with id: ") + std::to_string(kernelId));
    }

    std::vector<KernelArgument> referenceArguments = referenceClassResultMap.find(kernelId)->second;
    return validateArguments(resultArguments, referenceArguments);
}

bool ResultValidator::validateArgumentWithKernel(const size_t kernelId, const std::vector<KernelArgument>& resultArguments) const
{
    if (referenceKernelResultMap.find(kernelId) == referenceKernelResultMap.end())
    {
        throw std::runtime_error(std::string("No reference kernel results found for kernel with id: ") + std::to_string(kernelId));
    }

    std::vector<KernelArgument> referenceArguments = referenceKernelResultMap.find(kernelId)->second;
    return validateArguments(resultArguments, referenceArguments);
}

void ResultValidator::setReferenceClassResult(const size_t kernelId, const std::vector<KernelArgument>& classResult)
{
    if (referenceClassResultMap.find(kernelId) != referenceClassResultMap.end())
    {
        referenceClassResultMap.erase(kernelId);
    }
    referenceClassResultMap.insert(std::make_pair(kernelId, classResult));
}

void ResultValidator::setReferenceKernelResult(const size_t kernelId, const std::vector<KernelArgument>& kernelResult)
{
    if (referenceKernelResultMap.find(kernelId) != referenceKernelResultMap.end())
    {
        referenceKernelResultMap.erase(kernelId);
    }
    referenceKernelResultMap.insert(std::make_pair(kernelId, kernelResult));
}

bool ResultValidator::hasReferenceClassResult(const size_t kernelId) const
{
    return referenceClassResultMap.find(kernelId) != referenceClassResultMap.end();
}

bool ResultValidator::hasReferenceKernelResult(const size_t kernelId) const
{
    return referenceKernelResultMap.find(kernelId) != referenceKernelResultMap.end();
}

void ResultValidator::clearReferenceResults()
{
    referenceClassResultMap.clear();
    referenceKernelResultMap.clear();
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

double ResultValidator::getToleranceThreshold() const
{
    return toleranceThreshold;
}

ValidationMethod ResultValidator::getValidationMethod() const
{
    return validationMethod;
}

bool ResultValidator::validateArguments(const std::vector<KernelArgument>& resultArguments,
    const std::vector<KernelArgument>& referenceArguments) const
{
    bool validationResult = true;

    for (const auto& referenceArgument : referenceArguments)
    {
        ArgumentDataType referenceDataType = referenceArgument.getArgumentDataType();
        KernelArgument kernelArgument = findArgument(referenceArgument.getId(), resultArguments);

        if (referenceDataType != kernelArgument.getArgumentDataType())
        {
            throw std::runtime_error(std::string("Reference class argument data type mismatch for argument id: ")
                + std::to_string(kernelArgument.getId()));
        }

        if (referenceDataType == ArgumentDataType::Double)
        {
            validationResult &= validateResult(kernelArgument.getDataDouble(), referenceArgument.getDataDouble());
        }
        else if (referenceDataType == ArgumentDataType::Float)
        {
            validationResult &= validateResult(kernelArgument.getDataFloat(), referenceArgument.getDataFloat());
        }
        else if (referenceDataType == ArgumentDataType::Int)
        {
            validationResult &= validateResult(kernelArgument.getDataInt(), referenceArgument.getDataInt());
        }
        else if (referenceDataType == ArgumentDataType::Short)
        {
            validationResult &= validateResult(kernelArgument.getDataShort(), referenceArgument.getDataShort());
        }
        else
        {
            throw std::runtime_error("Unsupported argument data type");
        }
    }

    return validationResult;
}

KernelArgument ResultValidator::findArgument(const size_t argumentId, const std::vector<KernelArgument>& arguments) const
{
    for (const auto& argument : arguments)
    {
        if (argument.getId() == argumentId)
        {
            return argument;
        }
    }

    throw std::runtime_error(std::string("Reference kernel argument with following id is not associated with given kernel: ")
        + std::to_string(argumentId));
}

} // namespace ktt
