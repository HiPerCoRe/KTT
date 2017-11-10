/** @file tuner_api.h
  * @brief File containing public API for KTT library.
  */
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
#include "enum/argument_upload_type.h"
#include "enum/compute_api.h"
#include "enum/dimension.h"
#include "enum/global_size_type.h"
#include "enum/print_format.h"
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

/** @namespace ktt
  * @brief All classes, methods and type aliases related to KTT library are located inside ktt namespace.
  */
namespace ktt
{

using half_float::half;
class TunerCore;

/** @class Tuner
  * @brief Class which serves as the main part of public API for KTT library.
  */
class KTT_API Tuner
{
public:
    /** @fn Tuner(const size_t platformIndex, const size_t deviceIndex)
      * @brief Constructor, which creates new tuner object for specified platform and device. Tuner uses OpenCL as compute API. Indices for available
      * platforms and devices can be retrieved by calling printComputeApiInfo() method.
      * @param platformIndex Index for platform used by created tuner.
      * @param deviceIndex Index for device used by created tuner.
      */
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);

    /** @fn Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi)
      * @brief Constructor, which creates new tuner object for specified platform, device and compute API. Indices for available platforms
      * and devices can be retrieved by calling printComputeApiInfo() method. If specified compute API is CUDA, platform index is ignored.
      * @param platformIndex Index for platform used by created tuner.
      * @param deviceIndex Index for device used by created tuner.
      * @param computeApi Compute API used by created tuner.
      */
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi);

    /** @fn ~Tuner()
      * @brief Tuner destructor.
      */
    ~Tuner();

    /** @fn addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)
      * @brief Adds new kernel to tuner from source inside string. Requires specification of kernel name and default global and local thread sizes.
      * @param source Kernel source code written in corresponding compute API language.
      * @param kernelName Name of kernel function inside kernel source code.
      * @param globalSize Dimensions for base kernel global size (eg. grid size in CUDA).
      * @param localSize Dimensions for base kernel local size (eg. block size in CUDA).
      * @return Id assigned to kernel by tuner. The id can be used in other API methods.
      */
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    /** @fn addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
      * const DimensionVector& localSize)
      * @brief Adds new kernel to tuner from file. Requires specification of kernel name and default global and local thread sizes.
      * @param filePath Path to file with kernel source code written in corresponding compute API language.
      * @param kernelName Name of kernel function inside kernel source code.
      * @param globalSize Dimensions for base kernel global size (eg. grid size in CUDA).
      * @param localSize Dimensions for base kernel local size (eg. block size in CUDA).
      * @return Id assigned to kernel by tuner. The id can be used in other API methods.
      */
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    /** @fn setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
      * @brief Sets kernel arguments for specified kernel by providing corresponding argument ids.
      * @param id Id of kernel for which the arguments are set.
      * @param argumentIds Ids of arguments to be used by specified kernel. Order of ids must match the order of kernel arguments specified in kernel
      * function. Argument ids for single kernel must be unique.
      */
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);

    /** @fn addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
      * @brief Adds new parameter for specified kernel, providing parameter name and list of allowed values. When the corresponding kernel
      * is launched, parameters will be added to kernel source code as preprocessor definitions. During the tuning process, tuner will generate
      * configurations for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter is added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues List of allowed values for the parameter.
      */
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues);

    /** @fn addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
      * const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension)
      * @brief Adds new parameter for specified kernel, providing parameter name and list of allowed values. When the corresponding kernel
      * is launched, parameters will be added to kernel source code as preprocessor definitions. During the tuning process, tuner will generate
      * configurations for combinations of kernel parameters and their values.
      *
      * This version of method allows the parameter to act as thread size modifier. Parameter value modifies number of threads in either global
      * or local space in specified dimension. Form of modification depends on thread modifier action argument. If there are multiple thread
      * modifiers present for same space and dimension, actions are applied in the order of parameters' addition.
      * @param id Id of kernel for which the parameter is added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues List of allowed values for the parameter.
      * @param modifierType Type of thread modifier. See ThreadModifierType for more information.
      * @param modifierAction Action of thread modifier. See ThreadModifierAction for more information.
      * @param modifierDimension Dimension which will be affected by thread modifier. See Dimension for more information.
      */
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension);

    /** @fn addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
      * const std::vector<std::string>& parameterNames)
      * @brief Adds new constraint for specified kernel. Constraints are used to prevent generating of invalid
      * configurations (eg. conflicting parameter values).
      * @param id Id of kernel for which the constraint is added.
      * @param constraintFunction Function which returns true if provided combination of parameter values is valid. Returns false otherwise.
      * @param parameterNames Names of kernel parameters which will be affected by the constraint function. The order of parameter names will
      * correspond to the order of parameter values inside constraint function vector argument.
      */
    void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);

    /** @fn setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
      * @brief Sets tuning manipulator for specified kernel. Tuning manipulator enables customization of kernel execution. This is useful in several
      * cases, eg. running part of the computation in C++ code, utilizing iterative kernel launches or composite kernels. See TuningManipulator for
      * more information.
      * @param id Id of kernel for which the tuning manipulator is set.
      * @param manipulator Tuning manipulator for specified kernel.
      */
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    /** @fn addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
      * std::unique_ptr<TuningManipulator> manipulator)
      * @brief Creates a kernel composition using specified kernels. Following methods can be used with kernel compositions and will call
      * the corresponding method for all kernels inside the composition: setKernelArguments(), addParameter() (both versions), addConstraint().
      * 
      * Kernel compositions do not inherit any parameters or constraints from the original kernels. Setting kernel arguments and adding parameters
      * or constraints to kernels inside given composition will not affect the original kernels or other compositions. Tuning manipulator is required
      * in order to launch kernel composition with tuner. See TuningManipulator for more information.
      * @param compositionName Name of kernel composition. The name is used during output printing.
      * @param kernelIds Ids of kernels which will be included in the composition.
      * @param manipulator Tuning manipulator for the composition.
      * @return Id assigned to kernel composition by tuner. The id can be used in other API methods.
      */
    KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
        std::unique_ptr<TuningManipulator> manipulator);

    /** @fn addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
      * const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
      * const Dimension& modifierDimension)
      * @brief Calls addParameter() method (version with thread modifier) for a single kernel inside specified kernel composition. Does not affect
      * standalone kernels or other compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the parameter is added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues List of allowed values for the parameter.
      * @param modifierType Type of thread modifier. See ThreadModifierType for more information.
      * @param modifierAction Action of thread modifier. See ThreadModifierAction for more information.
      * @param modifierDimension Dimension which will be affected by thread modifier. See Dimension for more information.
      */
    void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
        const Dimension& modifierDimension);

    /** @fn setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
      * @brief Calls setKernelArguments() method for a single kernel inside specified kernel composition. Does not affect standalone kernels or other
      * compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the arguments are set.
      * @param argumentIds Ids of arguments to be used by specified kernel inside the composition. Order of ids must match the order of kernel
      * arguments specified in kernel function. Argument ids for single kernel must be unique.
      */
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    // Argument handling methods
    template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType& accessType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, ArgumentMemoryLocation::Device, accessType, ArgumentUploadType::Vector);
    }
    template <typename T> ArgumentId addArgumentVector(std::vector<T>& data, const ArgumentAccessType& accessType,
        const ArgumentMemoryLocation& memoryLocation, const bool copyData)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, memoryLocation, accessType, copyData);
    }
    template <typename T> ArgumentId addArgumentScalar(const T& data)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(&data, 1, sizeof(T), dataType, ArgumentMemoryLocation::Device, ArgumentAccessType::ReadOnly, ArgumentUploadType::Scalar);
    }
    template <typename T> ArgumentId addArgumentLocal(const size_t localMemoryElementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(localMemoryElementsCount, sizeof(T), dataType);
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
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);

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
    ArgumentId addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const bool copyData);
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType);
    ArgumentId addArgument(const size_t localMemoryElementsCount, const size_t elementSizeInBytes, const ArgumentDataType& dataType);

    template <typename T> ArgumentDataType getMatchingArgumentDataType() const
    {
        if (!std::is_trivially_copyable<T>() || typeid(T) == typeid(bool))
        {
            std::cerr << "Unsupported argument data type" << std::endl;
            throw std::runtime_error("Unsupported argument data type");
        }

        if (!std::is_arithmetic<T>())
        {
            return ArgumentDataType::Custom;
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

        std::cerr << "Unsupported argument data type" << std::endl;
        throw std::runtime_error("Unsupported argument data type");
    }
};

} // namespace ktt
