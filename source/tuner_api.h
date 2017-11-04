#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

// Compatibility for multiple platforms
#include "ktt_platform.h"

// Data types and enums
#include "ktt_types.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"
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

// Data holders
#include "api/argument_output_descriptor.h"
#include "api/device_info.h"
#include "api/dimension_vector.h"
#include "api/platform_info.h"

// Reference class interface
#include "api/reference_class.h"

// Tuning manipulator interface
#include "api/tuning_manipulator.h"

// Support for 16-bit floating point data type
#include "half.hpp"

namespace ktt
{

using half_float::half;
class TunerCore;

class KTT_API Tuner
{
public:
    // Constructors and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi);
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode);
    ~Tuner();

    // Basic kernel handling methods
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues);

    // Advanced kernel handling methods
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension);
    void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    // Composition handling methods
    KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
        std::unique_ptr<TuningManipulator> manipulator);
    void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
        const Dimension& modifierDimension);
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    // Argument handling methods
    template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType& accessType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), dataType, ArgumentMemoryLocation::Device, accessType);
    }
    template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType& accessType,
        const ArgumentMemoryLocation& memoryLocation)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), dataType, memoryLocation, accessType);
    }
    template <typename T> ArgumentId addArgumentScalar(const T& data)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(&data, dataType);
    }
    template <typename T> ArgumentId addArgumentLocal(const size_t localMemoryElementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(localMemoryElementsCount, dataType);
    }

    // Kernel launch and tuning methods
    void tuneKernel(const KernelId id);
    void tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output);
    void runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    void setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments);

    // Result retrieval methods
    void setPrintingTimeUnit(const TimeUnit& unit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const;
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const;
    std::vector<ParameterPair> getBestConfiguration(const KernelId id) const;

    // Result validation methods
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    void setValidationMethod(const ValidationMethod& method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;
    DeviceInfo getCurrentDeviceInfo() const;

    // Utility methods
    void setAutomaticGlobalSizeCorrection(const bool flag);
    void setGlobalSizeType(const GlobalSizeType& type);
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

private:
    // Pointer to implementation class
    std::unique_ptr<TunerCore> tunerCore;

    // Helper methods
    KernelId addArgument(const void* vectorData, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType);
    KernelId addArgument(const void* scalarData, const ArgumentDataType& dataType);
    KernelId addArgument(const size_t localMemoryElementsCount, const ArgumentDataType& dataType);

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
