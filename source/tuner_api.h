#pragma once

#ifndef KTT_API
#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251) // MSVC irrelevant warning (as long as there are no public attributes)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER
#endif // KTT_API

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"
#include "enum/argument_print_condition.h"
#include "enum/compute_api.h"
#include "enum/dimension.h"
#include "enum/global_size_type.h"
#include "enum/print_format.h"
#include "enum/run_mode.h"
#include "enum/time_unit.h"
#include "enum/search_method.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"
#include "enum/validation_method.h"

// Information about platforms and devices
#include "api/device_info.h"
#include "api/platform_info.h"

// Description of kernel output
#include "api/argument_output_descriptor.h"

// Reference class interface
#include "api/reference_class.h"

// Tuning manipulator interface
#include "api/tuning_manipulator.h"

// Support for 16-bit floating point data type
#include "half.hpp"

namespace ktt
{

using half_float::half; // Utilize half data type without namespace specifier

class TunerCore; // Forward declaration of TunerCore class

class KTT_API Tuner
{
public:
    // Constructors and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi);
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode);
    ~Tuner();

    // Basic kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues);

    // Advanced kernel handling methods
    void addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);

    // Composition handling methods
    size_t addKernelComposition(const std::string& compositionName, const std::vector<size_t>& kernelIds,
        std::unique_ptr<TuningManipulator> tuningManipulator);
    void addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction,
        const Dimension& modifierDimension);
    void setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds);

    // Argument handling methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentAccessType& accessType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), dataType, ArgumentMemoryLocation::Device, accessType);
    }
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentAccessType& accessType,
        const ArgumentMemoryLocation& memoryLocation)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), dataType, memoryLocation, accessType);
    }
    template <typename T> size_t addArgument(const T& scalarValue)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(&scalarValue, dataType);
    }
    template <typename T> size_t addArgument(const size_t localMemoryElementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(localMemoryElementsCount, dataType);
    }
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Kernel launch and tuning methods
    void tuneKernel(const size_t kernelId);
    void runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    void setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Result retrieval methods
    void setPrintingTimeUnit(const TimeUnit& timeUnit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;
    std::vector<ParameterValue> getBestConfiguration(const size_t kernelId) const;

    // Result validation methods
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setValidationRange(const size_t argumentId, const size_t validationRange);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;
    DeviceInfo getCurrentDeviceInfo() const;

    // Utility methods
    void setGlobalSizeType(const GlobalSizeType& globalSizeType);
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

private:
    // Pointer to implementation class
    std::unique_ptr<TunerCore> tunerCore;

    // Helper methods
    size_t addArgument(const void* vectorData, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType);
    size_t addArgument(const void* scalarData, const ArgumentDataType& dataType);
    size_t addArgument(const size_t localMemoryElementsCount, const ArgumentDataType& dataType);

    template <typename T> ArgumentDataType getMatchingArgumentDataType() const
    {
        if (!std::is_trivially_copyable<T>())
        {
            std::cerr << "Unsupported argument data type" << std::endl;
            throw std::runtime_error("Unsupported argument data type");
        }

        if (sizeof(T) == 1 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedChar;
        }
        else if (sizeof(T) == 1)
        {
            return ArgumentDataType::Char;
        }
        else if (typeid(T) == typeid(half))
        {
            return ArgumentDataType::Half;
        }
        else if (sizeof(T) == 2 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedShort;
        }
        else if (sizeof(T) == 2)
        {
            return ArgumentDataType::Short;
        }
        else if (typeid(T) == typeid(float))
        {
            return ArgumentDataType::Float;
        }
        else if (sizeof(T) == 4 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedInt;
        }
        else if (sizeof(T) == 4)
        {
            return ArgumentDataType::Int;
        }
        else if (typeid(T) == typeid(double))
        {
            return ArgumentDataType::Double;
        }
        else if (sizeof(T) == 8 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedLong;
        }
        else if (sizeof(T) == 8)
        {
            return ArgumentDataType::Long;
        }
        else
        {
            std::cerr << "Unsupported argument data type" << std::endl;
            throw std::runtime_error("Unsupported argument data type");
        }
    }
};

} // namespace ktt
