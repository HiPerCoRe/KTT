#include "result_validator.h"

namespace ktt
{

ResultValidator::ResultValidator():
    toleranceThreshold(0.0001),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

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

template <typename T> bool ResultValidator::validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult) const
{
    return false;
}

} // namespace ktt
