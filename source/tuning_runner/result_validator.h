#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "api/reference_class.h"
#include "compute_engine/compute_engine.h"
#include "enum/validation_method.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "utility/argument_printer.h"
#include "utility/logger.h"

namespace ktt
{

class ResultValidator
{
public:
    // Constructor
    explicit ResultValidator(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine);

    // Core methods
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setToleranceThreshold(const double toleranceThreshold);
    void setValidationMethod(const ValidationMethod& validationMethod);
    void setValidationRange(const size_t argumentId, const size_t validationRange);
    void computeReferenceResult(const Kernel* kernel);
    void clearReferenceResults();
    bool validateArgumentsWithClass(const Kernel* kernel, const KernelConfiguration& kernelConfiguration);
    bool validateArgumentsWithKernel(const Kernel* kernel, const KernelConfiguration& kernelConfiguration);
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Getters
    double getToleranceThreshold() const;
    ValidationMethod getValidationMethod() const;

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeEngine* computeEngine;
    ArgumentPrinter argumentPrinter;
    double toleranceThreshold;
    ValidationMethod validationMethod;
    std::map<size_t, size_t> argumentValidationRangeMap;
    std::map<size_t, std::tuple<std::unique_ptr<ReferenceClass>, std::vector<size_t>>> referenceClassMap;
    std::map<size_t, std::tuple<size_t, std::vector<ParameterValue>, std::vector<size_t>>> referenceKernelMap;
    std::map<size_t, std::vector<KernelArgument>> referenceClassResultMap;
    std::map<size_t, std::vector<KernelArgument>> referenceKernelResultMap;

    // Helper methods
    void computeReferenceResultWithClass(const Kernel* kernel);
    void computeReferenceResultWithKernel(const Kernel* kernel);
    bool validateArguments(const std::vector<KernelArgument>& resultArguments, const std::vector<KernelArgument>& referenceArguments,
        const std::string kernelName, const KernelConfiguration& kernelConfiguration) const;
    std::vector<const KernelArgument*> getKernelArgumentPointers(const size_t kernelId) const;

    template <typename T> bool validateResult(const std::vector<T>& result, const std::vector<T>& referenceResult, const size_t argumentId) const
    {
        auto argumentRangePointer = argumentValidationRangeMap.find(argumentId);
        if (argumentRangePointer == argumentValidationRangeMap.end() && result.size() != referenceResult.size())
        {
            logger->log(std::string("Number of elements in results differs for argument with id: ") + std::to_string(argumentId)
                + ", reference size: " + std::to_string(referenceResult.size()) + ", result size: " + std::to_string(result.size()));
            return false;
        }
        if (argumentRangePointer != argumentValidationRangeMap.end())
        {
            return validateResultInner(result, referenceResult, argumentRangePointer->second, argumentId, std::is_floating_point<T>());
        }
        return validateResultInner(result, referenceResult, referenceResult.size(), argumentId, std::is_floating_point<T>());
    }

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult, const size_t range,
        const size_t argumentId, std::true_type) const
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
                logger->log(std::string("Results differ for argument with id: ") + std::to_string(argumentId) + ", absolute difference is: "
                    + std::to_string(difference));
                return false;
            }
            return true;
        }
        else
        {
            for (size_t i = 0; i < range; i++)
            {
                if (std::fabs(result.at(i) - referenceResult.at(i)) > toleranceThreshold)
                {
                    logger->log(std::string("Results differ for argument with id: ") + std::to_string(argumentId) + ", index: " + std::to_string(i)
                        + ", reference value: " + std::to_string(referenceResult.at(i)) + ", result value: " + std::to_string(result.at(i))
                        + ", difference: " + std::to_string(std::fabs(result.at(i) - referenceResult.at(i))));
                    return false;
                }
            }
            return true;
        }
    }

    template <typename T> bool validateResultInner(const std::vector<T>& result, const std::vector<T>& referenceResult, const size_t range,
        const size_t argumentId, std::false_type) const
    {
        for (size_t i = 0; i < range; i++)
        {
            if (result.at(i) != referenceResult.at(i))
            {
                logger->log(std::string("Results differ for argument with id: ") + std::to_string(argumentId) + ", index: " + std::to_string(i)
                    + ", reference value: " + std::to_string(referenceResult.at(i)) + ", result value: " + std::to_string(result.at(i))
                    + ", difference: " + std::to_string(std::fabs(result.at(i) - referenceResult.at(i))));
                return false;
            }
        }
        return true;
    }
};

} // namespace ktt
