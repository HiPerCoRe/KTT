#include "result_validator.h"

namespace ktt
{

ResultValidator::ResultValidator():
    toleranceThreshold(0.0001),
    validationMethod(ValidationMethod::SideBySideComparison)
{}

template <typename T> bool ResultValidator::validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult) const
{
    return false;
}

} // namespace ktt
