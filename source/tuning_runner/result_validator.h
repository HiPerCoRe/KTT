#pragma once

#include <vector>

#include "../enum/validation_method.h"

namespace ktt
{

class ResultValidator
{
public:
    ResultValidator();

    template <typename T> bool validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult) const;

private:
    double toleranceThreshold;
    ValidationMethod validationMethod;
};

} // namespace ktt
