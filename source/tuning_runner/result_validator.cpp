#include "result_validator.h"

namespace ktt
{

ResultValidator::ResultValidator():
    toleranceThreshold(1e-4),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

// WIP
bool ResultValidator::validateResultWithClass(const size_t kernelId, const KernelArgument& kernelArgument) const
{
    return true;
}

bool ResultValidator::validateResultWithKernel(const size_t kernelId, const std::vector<KernelArgument>& resultArguments) const
{
    return true;
}

void ResultValidator::setReferenceClassArgument(const size_t kernelId, const KernelArgument& kernelArgument)
{

}

void ResultValidator::setReferenceKernelArguments(const size_t kernelId, const std::vector<KernelArgument>& kernelArguments)
{

}

bool ResultValidator::hasReferenceClassArgument(const size_t kernelId) const
{
    return false;
}

bool ResultValidator::hasReferenceKernelArguments(const size_t kernelId) const
{
    return false;
}

void ResultValidator::clearReferenceArguments(const size_t kernelId)
{

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

} // namespace ktt
