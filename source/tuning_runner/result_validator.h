#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>

#include "../enum/validation_method.h"
#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class ResultValidator
{
public:
    // Constructor
    ResultValidator();

    // Core methods
    bool validateResultWithClass(const size_t kernelId, const KernelArgument& kernelArgument) const;
    bool validateResultWithKernel(const size_t kernelId, const std::vector<KernelArgument>& resultArguments) const;
    void setReferenceClassArgument(const size_t kernelId, const KernelArgument& kernelArgument);
    void setReferenceKernelArguments(const size_t kernelId, const std::vector<KernelArgument>& kernelArguments);
    bool hasReferenceClassArgument(const size_t kernelId) const;
    bool hasReferenceKernelArguments(const size_t kernelId) const;
    void clearReferenceArguments(const size_t kernelId);

    // Setters
    void setToleranceThreshold(const double toleranceThreshold);
    void setValidationMethod(const ValidationMethod& validationMethod);

    // Getters
    double getToleranceThreshold() const;
    ValidationMethod getValidationMethod() const;

private:
    // Attributes
    double toleranceThreshold;
    ValidationMethod validationMethod;
    std::map<size_t, KernelArgument> referenceClassArgumentMap;
    std::map<size_t, std::vector<KernelArgument>> referenceKernelArgumentMap;

    // Helper methods
    template <typename T> bool validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult) const
    {
        if (result.size() != referenceResult.size())
        {
            std::cerr << "Number of elements in results differs, reference size: <" << referenceResult.size() << ">; result size: <" << result.size()
                << ">" << std::endl;
            return false;
        }
        validateResultInner(result, referenceResult, std::is_floating_point<T>());
    }

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult,
        std::true_type) const
    {
        if (validationMethod == ValidationMethod::AbsoluteDifference)
        {
            double difference = 0.0;
            for (size_t i = 0; i < result.size(); i++)
            {
                difference += std::fabs(result.at(i) - referenceResult.at(i));
            }
            if (difference > toleranceThreshold)
            {
                std::cerr << "Results differ, absolute difference value: <" << difference << ">" << std::endl;
                return false;
            }
            return true;
        }
        else
        {
            for (size_t i = 0; i < result.size(); i++)
            {
                if (std::fabs(result.at(i) - referenceResult.at(i)) > toleranceThreshold)
                {
                    std::cerr << "Results differ at index <" << i << ">; reference value: <" << referenceResult.at(i) << ">; result value: <"
                        << result.at(i) << ">" << std::endl;
                    return false;
                }
            }
            return true;
        }
    }

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult,
        std::false_type) const
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            if (result.at(i) != referenceResult.at(i))
            {
                std::cerr << "Results differ at index <" << i << ">; reference value: <" << referenceResult.at(i) << ">; result value: <"
                    << result.at(i) << ">" << std::endl;
                return false;
            }
        }
        return true;
    }
};

} // namespace ktt
