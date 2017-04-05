#pragma once

#include <vector>

#include "../enum/validation_method.h"

namespace ktt
{

class ResultValidator
{
public:
    ResultValidator();

    void setToleranceThreshold(const double toleranceThreshold);
    void setValidationMethod(const ValidationMethod& validationMethod);

    double getToleranceThreshold() const;
    ValidationMethod getValidationMethod() const;

    template <typename T> bool validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult) const;

private:
    double toleranceThreshold;
    ValidationMethod validationMethod;
};

} // namespace ktt
