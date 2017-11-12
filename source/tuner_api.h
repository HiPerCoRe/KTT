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
    /** @fn explicit Tuner(const size_t platformIndex, const size_t deviceIndex)
      * @brief Constructor, which creates new tuner object for specified platform and device. Tuner uses OpenCL as compute API. Indices for available
      * platforms and devices can be retrieved by calling printComputeApiInfo() method.
      * @param platformIndex Index for platform used by created tuner.
      * @param deviceIndex Index for device used by created tuner.
      */
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);

    /** @fn explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi)
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

    /** @fn KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
      * const DimensionVector& localSize)
      * @brief Adds new kernel to tuner from source inside string. Requires specification of kernel name and default global and local thread sizes.
      * @param source Kernel source code written in corresponding compute API language.
      * @param kernelName Name of kernel function inside kernel source code.
      * @param globalSize Dimensions for base kernel global size (eg. grid size in CUDA).
      * @param localSize Dimensions for base kernel local size (eg. block size in CUDA).
      * @return Id assigned to kernel by tuner. The id can be used in other API methods.
      */
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    /** @fn KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
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

    /** @fn void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
      * @brief Sets kernel arguments for specified kernel by providing corresponding argument ids.
      * @param id Id of kernel for which the arguments are set.
      * @param argumentIds Ids of arguments to be used by specified kernel. Order of ids must match the order of kernel arguments specified in kernel
      * function. Argument ids for single kernel must be unique.
      */
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);

    /** @fn void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
      * @brief Adds new parameter for specified kernel, providing parameter name and list of allowed values. When the corresponding kernel
      * is launched, parameters will be added to kernel source code as preprocessor definitions. During the tuning process, tuner will generate
      * configurations for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter is added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues List of allowed values for the parameter.
      */
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues);

    /** @fn void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
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

    /** @fn void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
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

    /** @fn void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
      * @brief Sets tuning manipulator for specified kernel. Tuning manipulator enables customization of kernel execution. This is useful in several
      * cases, eg. running part of the computation in C++ code, utilizing iterative kernel launches or composite kernels. See TuningManipulator for
      * more information.
      * @param id Id of kernel for which the tuning manipulator is set.
      * @param manipulator Tuning manipulator for specified kernel.
      */
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    /** @fn KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
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

    /** @fn void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
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

    /** @fn void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
      * @brief Calls setKernelArguments() method for a single kernel inside specified kernel composition. Does not affect standalone kernels or other
      * compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the arguments are set.
      * @param argumentIds Ids of arguments to be used by specified kernel inside the composition. Order of ids must match the order of kernel
      * arguments specified in kernel function. Argument ids for single kernel must be unique.
      */
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    /** @fn template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType& accessType)
      * @brief Adds new vector argument to tuner. Makes copy of argument data, so the source data vector remains unaffected by tuner operations.
      * Argument data will be accessed from device memory during its usage by compute API.
      * @param data Argument data provided in std::vector. Provided data type must be trivially copyable. Bool data type is currently not supported.
      * @param accessType Access type of argument specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType& accessType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, ArgumentMemoryLocation::Device, accessType, ArgumentUploadType::Vector);
    }

    /** @fn template <typename T> ArgumentId addArgumentVector(std::vector<T>& data, const ArgumentAccessType& accessType,
      * const ArgumentMemoryLocation& memoryLocation, const bool copyData)
      * @brief Adds new vector argument to tuner. Allows choice for argument memory location and whether argument data is copied to tuner.
      * @param data Argument data provided in std::vector. Provided data type must be trivially copyable. Bool data type is currently not supported.
      * @param accessType Access type of argument specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @param memoryLocation Memory location of argument specifies whether argument will be accessed from device or host memory during its usage
      * by compute API. See ArgumentMemoryLocation for more information.
      * @param copyData Flag which specifies whether the argument is copied inside tuner. If set to false, tuner will store reference of source data
      * vector and will access it directly during kernel launch operations. This results in lower memory overhead, but relies on a user to keep data
      * in source vector valid.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentVector(std::vector<T>& data, const ArgumentAccessType& accessType,
        const ArgumentMemoryLocation& memoryLocation, const bool copyData)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, memoryLocation, accessType, copyData);
    }

    /** @fn template <typename T> ArgumentId addArgumentScalar(const T& data)
      * @brief Adds new scalar argument to tuner. All scalar arguments are read-only.
      * @param data Argument data provided as single scalar value. The data type must be trivially copyable. Bool data type is currently not
      * supported.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentScalar(const T& data)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(&data, 1, sizeof(T), dataType, ArgumentMemoryLocation::Device, ArgumentAccessType::ReadOnly, ArgumentUploadType::Scalar);
    }

    /** @fn template <typename T> ArgumentId addArgumentLocal(const size_t localMemoryElementsCount)
      * @brief Adds new local memory (shared memory in CUDA) argument to tuner. All local memory arguments are read-only.
      * @param localMemoryElementsCount Specifies how many elements of provided data type the argument contains.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentLocal(const size_t localMemoryElementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(localMemoryElementsCount, sizeof(T), dataType);
    }

    /** @fn void tuneKernel(const KernelId id)
      * @brief Starts the tuning process for specified kernel. Creates configuration space based on combinations of provided kernel parameters
      * and constraints. The configurations will be launched in order that depends on specified SearchMethod.
      * @param id Id of kernel for which the tuning begins.
      */
    void tuneKernel(const KernelId id);

    /** @fn void tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
      * @brief Performs one step of the tuning process for specified kernel. When this method is called inside tuner for the first time, creates
      * configuration space based on combinations of provided kernel parameters and constraints. Each time this method is called, launches single
      * kernel configuration. If all configurations were already tested, runs kernel using the best configuration. Output data can be retrieved
      * by providing output descriptors.
      * @param id Id of kernel for which the tuning by step begins.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See ArgumentOutputDescriptor for more
      * information.
      */
    void tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output);

    /** @fn void runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output)
      * @brief Runs specified kernel using provided configuration. Does not perform result validation.
      * @param id Id of kernel which is run.
      * @param configuration Configuration under which the kernel will be launched. See ::ParameterPair for more information.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See ArgumentOutputDescriptor for more
      * information.
      */
    void runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);

    /** @fn void setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments)
      * @brief Specifies search method which will be used during kernel tuning. Number of required search arguments depends on the search method.
      * Default search method is full search, which requires no search arguments.
      * @param method Search method which will be used during kernel tuning. See SearchMethod for more information.
      * @param arguments Arguments necessary for specified search method to work. Following arguments are required for corresponding search method,
      * the order of arguments is important:
      * - RandomSearch - fraction
      * - PSO - fraction, swarm size, global influence, local influence, random influence
      * - Annealing - fraction, maximum temperature
      * 
      * Fraction argument specifies the number of configurations which will be explored, eg. when fraction is set to 0.5, 50% of all configurations
      * will be explored.
      */
    void setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments);

    /** @fn void setPrintingTimeUnit(const TimeUnit& unit)
      * @brief Sets time unit used during printing of results inside printResult() methods. Default time unit is microseconds. 
      * @param unit Time unit which will be used inside printResult() methods. See TimeUnit for more information.
      */
    void setPrintingTimeUnit(const TimeUnit& unit);

    /** @fn void setInvalidResultPrinting(const bool flag)
      * @brief Toggles printing of results from failed kernel runs. Invalid results will be separated from valid results during printing.
      * Printing of invalid results is disabled by default.
      * @param flag If true, printing of invalid results is enabled. It is disabled otherwise.
      */
    void setInvalidResultPrinting(const bool flag);

    /** @fn void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const
      * @brief Prints tuning results for specified kernel to specified output stream. Valid results will be printed only if methods tuneKernel() or
      * tuneKernelByStep() were already called for corresponding kernel.
      * @param id Id of kernel for which the results are printed.
      * @param outputTarget Location where the results are printed.
      * @param format Format in which the results are printed. See PrintFormat for more information.
      */
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const;

    /** @fn void printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const
      * @brief Prints tuning results for specified kernel to specified file. Valid results will be printed only if methods tuneKernel() or
      * tuneKernelByStep() were already called for corresponding kernel.
      * @param id Id of kernel for which the results are printed.
      * @param filePath Path to file where the results are printed.
      * @param format Format in which the results are printed. See PrintFormat for more information.
      */
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const;

    /** @fn std::vector<ParameterPair> getBestConfiguration(const KernelId id) const
      * @brief Returns the best configuration found for specified kernel. Valid configuration will be returned only if methods tuneKernel() or
      * tuneKernelByStep() were already called for corresponding kernel.
      * @param id Id of kernel for which the best configuration is returned.
      * @return Best configuration found for specified kernel. See ::ParameterPair for more information.
      */
    std::vector<ParameterPair> getBestConfiguration(const KernelId id) const;

    /** @fn void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
      * const std::vector<ArgumentId>& validatedArgumentIds)
      * @brief Sets reference kernel for specified kernel. Reference kernel output will be compared to tuned kernel output in order to ensure
      * correctness of computation. Reference kernel uses only single configuration which cannot be composite and cannot use tuning manipulator.
      * @param id Id of kernel for which reference kernel is set.
      * @param referenceId Id of reference kernel. This can be the same as validated kernel. This can be useful in cases where kernel has
      * a configuration which is known to produce correct results.
      * @param referenceConfiguration Configuration under which the reference kernel will be launched to produce reference output.
      * @param validatedArgumentIds Ids of kernel arguments which will be validated. The validated arguments must be vector arguments and cannot be
      * read-only.
      */
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);

    /** @fn void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
      * const std::vector<ArgumentId>& validatedArgumentIds)
      * @brief Sets reference class for specified kernel. Reference class output will be compared to tuned kernel output in order to ensure
      * correctness of computation.
      * @param id Id of kernel for which reference class is set.
      * @param referenceClass Reference class which produces reference output for specified kernel. See ReferenceClass for more information.
      * @param validatedArgumentIds Ids of kernel arguments which will be validated. The validated arguments must be vector arguments and cannot be
      * read-only.
      */
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);

    /** @fn void setValidationMethod(const ValidationMethod& method, const double toleranceThreshold)
      * @brief Sets validation method and tolerance threshold for floating-point argument validation. Default validation method is side by side
      * comparison. Default tolerance threshold is 1e-4.
      * @param method Validation method which will be used for floating-point argument validation. See ValidationMethod for more information.
      * @param toleranceThreshold Output validation threshold. If difference between tuned kernel output and reference output is within tolerance
      * threshold, the tuned kernel output will be considered correct.
      */
    void setValidationMethod(const ValidationMethod& method, const double toleranceThreshold);

    /** @fn void setValidationRange(const ArgumentId id, const size_t range)
      * @brief Sets validation range for specified argument to specified validation range. Only elements within validation range, starting with
      * the first element, will be validated. All elements are validated by default.
      * @param id Id of argument for which the validation range is set.
      * @param range Range inside which the argument elements will be validated, starting from the first element.
      */
    void setValidationRange(const ArgumentId id, const size_t range);

    /** @fn void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
      * @brief Sets argument comparator for specified argument with custom data type. Arguments with custom data type cannot be compared using
      * built-in comparison operators and require user to provide comparator. Comparator cannot be set for built-in data types.
      * @param id Id of argument for which the comparator is set.
      * @param comparator Function which receives two elements with data type matching the data type of specified kernel argument and returns true
      * if the elements are equal. Returns false otherwise.
      */
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);

    /** @fn void setCompilerOptions(const std::string& options)
      * @brief Sets compute API compiler options to specified options. There are no default options for OpenCL back-end. Default option for CUDA
      * back-end is "--gpu-architecture=compute_30".
      * @param options Compute API compiler options. If multiple options are used, they need to be separated by a single space character.
      */
    void setCompilerOptions(const std::string& options);

    /** @fn void printComputeApiInfo(std::ostream& outputTarget) const
      * @brief Prints basic information about available platforms and devices to specified output stream. Also prints indices assigned to them
      * by KTT library.
      * @param outputTarget Location where the information is printed.
      */
    void printComputeApiInfo(std::ostream& outputTarget) const;

    /** @fn std::vector<PlatformInfo> getPlatformInfo() const
      * @brief Retrieves detailed information about all available platforms (eg. platform name, vendor). See PlatformInfo for more information.
      * @return Information about all available platforms.
      */
    std::vector<PlatformInfo> getPlatformInfo() const;

    /** @fn std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const
      * @brief Retrieves detailed information about all available devices (eg. device name, memory capacity) on specified platform. See DeviceInfo
      * for more information.
      * @param platformIndex Index of platform for which the device information is retrieved.
      * @return Information about all available devices on specified platform.
      */
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;

    /** @fn DeviceInfo getCurrentDeviceInfo() const
      * @brief Retrieves detailed information about device (eg. device name, memory capacity) used by the tuner. See DeviceInfo for more information.
      * @return Information about device used by the tuner.
      */
    DeviceInfo getCurrentDeviceInfo() const;

    /** @fn void setAutomaticGlobalSizeCorrection(const bool flag)
      * @brief Toggles automatic correction for global size, which ensures that global size in each dimension is always a multiple of local size in
      * corresponding dimension. Performs a roundup to the nearest higher multiple. Automatic global size correction is disabled by default.
      * @param flag If true, automatic global size correction is enabled. It is disabled otherwise.
      */
    void setAutomaticGlobalSizeCorrection(const bool flag);

    /** @fn void setGlobalSizeType(const GlobalSizeType& type)
      * @brief Sets global size specification type to specified compute API style. In OpenCL, NDrange size is specified as number of work-items in
      * a work-group multiplied by number of work-groups. In CUDA, grid size is specified as number of threads in a block divided by number
      * of blocks. This method makes it possible to use OpenCL style in CUDA and vice versa. Default global size type is the one corresponding to
      * compute API of the tuner.
      * @param type Global size type which is set for tuner. See GlobalSizeType for more information.
      */
    void setGlobalSizeType(const GlobalSizeType& type);

    /** @fn void setLoggingTarget(std::ostream& outputTarget)
      * @brief Sets the target for info messages logging to specified output stream. Default logging target is `std::clog`.
      * @param outputTarget Location where tuner info messages are printed.
      */
    void setLoggingTarget(std::ostream& outputTarget);

    /** @fn void setLoggingTarget(const std::string& filePath)
      * @brief Sets the target for info messages logging to specified file. Default logging target is `std::clog`.
      * @param filePath Path to file where tuner info messages are printed.
      */
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
