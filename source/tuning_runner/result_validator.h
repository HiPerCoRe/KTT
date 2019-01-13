#pragma once

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

    template <typename T> bool validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult, const ArgumentId id) const
    {
        auto argumentRangePointer = argumentValidationRanges.find(id);
        if (argumentRangePointer == argumentValidationRanges.end() && result.size() != referenceResult.size())
        {
            Logger::getLogger().log(LoggingLevel::Warning, std::string("Number of elements in results differs for argument with id: ")
                + std::to_string(id) + ", reference size: " + std::to_string(referenceResult.size()) + ", result size: "
                + std::to_string(result.size()));
            return false;
        }
        if (argumentRangePointer != argumentValidationRanges.end())
        {
            return validateResultInner(result, referenceResult, argumentRangePointer->second, id, std::is_floating_point<T>());
        }
        return validateResultInner(result, referenceResult, referenceResult.size(), id, std::is_floating_point<T>());
    }

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult, const size_t range,
        const ArgumentId id, std::true_type) const
    {
        if (validationMethod == ValidationMethod::AbsoluteDifference)
        {
            double difference = 0.0;
            for (size_t i = 0; i < range; i++)
            {
                difference += std::fabs(result.at(i) - referenceResult.at(i));
            }
            if (difference > toleranceThreshold)
            {
                Logger::getLogger().log(LoggingLevel::Warning, std::string("Results differ for argument with id: ") + std::to_string(id)
                    + ", absolute difference is: " + std::to_string(difference));
                return false;
            }
            return true;
        }
        else if (validationMethod == ValidationMethod::SideBySideComparison)
        {
            for (size_t i = 0; i < range; i++)
            {
                if (std::fabs(result.at(i) - referenceResult.at(i)) > toleranceThreshold)
                {
                    Logger::getLogger().log(LoggingLevel::Warning, std::string("Results differ for argument with id: ") + std::to_string(id)
                        + ", index: " + std::to_string(i) + ", reference value: " + std::to_string(referenceResult.at(i)) + ", result value: "
                        + std::to_string(result.at(i)) + ", difference: " + std::to_string(std::fabs(result.at(i) - referenceResult.at(i))));
                    return false;
                }
            }
            return true;
        }
        else if (validationMethod == ValidationMethod::SideBySideRelativeComparison)
        {
            for (size_t i = 0; i < range; i++)
            {
                double difference = std::fabs(result.at(i) - referenceResult.at(i));
                if ((difference > 1e-4) && (difference / referenceResult.at(i) > toleranceThreshold))
                {
                    Logger::getLogger().log(LoggingLevel::Warning, std::string("Results differ for argument with id: ") + std::to_string(id)
                        + ", index: " + std::to_string(i) + ", reference value: " + std::to_string(referenceResult.at(i)) + ", result value: "
                        + std::to_string(result.at(i)) + ", relative difference: " + std::to_string(difference / referenceResult.at(i)));
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

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult, const size_t range,
        const ArgumentId id, std::false_type) const
    {
        for (size_t i = 0; i < range; i++)
        {
            if (result.at(i) != referenceResult.at(i))
            {
                Logger::getLogger().log(LoggingLevel::Warning, std::string("Results differ for argument with id: ") + std::to_string(id)
                    + ", index: " + std::to_string(i) + ", reference value: " + std::to_string(referenceResult.at(i)) + ", result value: "
                    + std::to_string(result.at(i)) + ", difference: "
                    + std::to_string(static_cast<T>(std::fabs(result.at(i) - referenceResult.at(i)))));
                return false;
            }
        }
        return true;
    }
};

} // namespace ktt
