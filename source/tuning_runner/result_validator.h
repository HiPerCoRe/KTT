#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <api/reference_class.h>
#include <enum/kernel_run_mode.h>
#include <enum/validation_method.h>
#include <enum/validation_mode.h>
#include <kernel/kernel_manager.h>
#include <kernel_argument/argument_manager.h>
#include <utility/logger.h>
#include <half.hpp>

namespace ktt
{

using half_float::half;

class KernelRunner; // Forward declaration in order to avoid cyclical header dependency

class ResultValidator
{
public:
    // Constructor
    explicit ResultValidator(ArgumentManager* argumentManager, KernelRunner* kernelRunner);

    // Core methods
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    void setToleranceThreshold(const double threshold);
    void setValidationMethod(const ValidationMethod method);
    void setValidationMode(const ValidationMode mode);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void computeReferenceResult(const Kernel& kernel, const KernelRunMode runMode);
    void clearReferenceResults();
    void clearReferenceResults(const KernelId id);
    bool validateArguments(const Kernel& kernel, const KernelRunMode runMode);
    bool validateArgumentsWithClass(const Kernel& kernel, const KernelRunMode runMode);
    bool validateArgumentsWithKernel(const Kernel& kernel, const KernelRunMode runMode);

    // Getters
    double getToleranceThreshold() const;
    ValidationMethod getValidationMethod() const;
    bool hasReferenceResult(const KernelId id) const;

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelRunner* kernelRunner;
    double toleranceThreshold;
    ValidationMethod validationMethod;
    ValidationMode validationMode;
    std::map<ArgumentId, size_t> argumentValidationRanges;
    std::map<ArgumentId, std::function<bool(const void*, const void*)>> argumentComparators;
    std::map<KernelId, std::tuple<std::unique_ptr<ReferenceClass>, std::vector<ArgumentId>>> referenceClasses;
    std::map<KernelId, std::tuple<KernelId, std::vector<ParameterPair>, std::vector<ArgumentId>>> referenceKernels;
    std::map<KernelId, std::vector<KernelArgument>> referenceClassResults;
    std::map<KernelId, std::vector<KernelArgument>> referenceKernelResults;

    // Helper methods
    void computeReferenceResultWithClass(const Kernel& kernel);
    void computeReferenceResultWithKernel(const Kernel& kernel);
    bool validateArguments(const std::vector<KernelArgument>& resultArguments, const std::vector<KernelArgument>& referenceArguments,
        const std::string kernelName) const;
    bool validateResultCustom(const ArgumentId id, const void* result, const void* referenceResult, const size_t numberOfElements,
        const size_t elementSizeInBytes, const std::function<bool(const void*, const void*)>& comparator) const;
    bool isRunModeValidated(const KernelRunMode mode);

    template <typename T>
    bool validateResult(const KernelArgument& resultArgument, const KernelArgument& referenceArgument, const ArgumentId id) const
    {
        auto argumentRangePointer = argumentValidationRanges.find(id);
        const size_t resultSize = resultArgument.getNumberOfElementsWithType<T>();
        const size_t referenceSize = referenceArgument.getNumberOfElementsWithType<T>();

        if (argumentRangePointer == argumentValidationRanges.end() && resultSize != referenceSize)
        {
            Logger::logWarning(std::string("Number of elements in results differs for argument with id: ") + std::to_string(id)
                + ", reference size: " + std::to_string(referenceSize) + ", result size: " + std::to_string(resultSize));
            return false;
        }

        const T* result = resultArgument.getDataWithType<T>();
        const T* reference = referenceArgument.getDataWithType<T>();
        size_t validationRange = referenceSize;

        if (argumentRangePointer != argumentValidationRanges.end())
        {
            validationRange = argumentRangePointer->second;

            if (validationRange > referenceSize || validationRange > resultSize)
            {
                validationRange = std::min(referenceSize, resultSize);
                Logger::logWarning(std::string("Specified validation range (") + std::to_string(argumentRangePointer->second)
                    + ") for argument with id: " + std::to_string(id) + " is larger than argument size. It was clamped to "
                    + std::to_string(validationRange));
            }
        }

        return validateResultInner(result, reference, validationRange, id, std::is_floating_point<T>());
    }

    template <typename T>
    bool validateResultInner(const T* result, const T* reference, const size_t range, const ArgumentId id, std::true_type) const
    {
        if (validationMethod == ValidationMethod::AbsoluteDifference)
        {
            double difference = 0.0;

            for (size_t i = 0; i < range; ++i)
            {
                difference += std::fabs(result[i] - reference[i]);
            }

            if (difference > toleranceThreshold)
            {
                Logger::logWarning(std::string("Results differ for argument with id: ") + std::to_string(id) + ", absolute difference is: "
                    + std::to_string(difference));
                return false;
            }

            return true;
        }
        else if (validationMethod == ValidationMethod::SideBySideComparison)
        {
            for (size_t i = 0; i < range; ++i)
            {
                if (std::fabs(result[i] - reference[i]) > toleranceThreshold)
                {
                    Logger::logWarning(std::string("Results differ for argument with id: ") + std::to_string(id) + ", index: " + std::to_string(i)
                        + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i]) + ", difference: "
                        + std::to_string(std::fabs(result[i] - reference[i])));
                    return false;
                }
            }
            return true;
        }
        else if (validationMethod == ValidationMethod::SideBySideRelativeComparison)
        {
            for (size_t i = 0; i < range; ++i)
            {
                double difference = std::fabs(result[i] - reference[i]);

                if ((difference > 1e-4) && (difference / reference[i] > toleranceThreshold))
                {
                    Logger::logWarning(std::string("Results differ for argument with id: ") + std::to_string(id) + ", index: " + std::to_string(i)
                        + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i])
                        + ", relative difference: " + std::to_string(difference / reference[i]));
                    return false;
                }
            }
            return true;
        }
        else
        {
            throw std::runtime_error("Unsupported validation method");
        }
    }

    template <typename T>
    bool validateResultInner(const T* result, const T* reference, const size_t range, const ArgumentId id, std::false_type) const
    {
        for (size_t i = 0; i < range; ++i)
        {
            if (result[i] != reference[i])
            {
                Logger::logWarning(std::string("Results differ for argument with id: ") + std::to_string(id) + ", index: " + std::to_string(i)
                    + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i]) + ", difference: "
                    + std::to_string(static_cast<T>(std::fabs(result[i] - reference[i]))));
                return false;
            }
        }
        return true;
    }
};

} // namespace ktt
