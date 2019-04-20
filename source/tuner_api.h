/** @file tuner_api.h
  * Public API of KTT framework.
  */
#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <vector>

// Compatibility for multiple platforms
#include <ktt_platform.h>

// Data types and enums
#include <ktt_types.h>
#include <enum/argument_access_type.h>
#include <enum/argument_data_type.h>
#include <enum/argument_memory_location.h>
#include <enum/argument_upload_type.h>
#include <enum/compute_api.h>
#include <enum/global_size_type.h>
#include <enum/logging_level.h>
#include <enum/modifier_action.h>
#include <enum/modifier_dimension.h>
#include <enum/modifier_type.h>
#include <enum/print_format.h>
#include <enum/time_unit.h>
#include <enum/search_method.h>
#include <enum/validation_method.h>
#include <enum/validation_mode.h>

// Data holders
#include <api/computation_result.h>
#include <api/device_info.h>
#include <api/dimension_vector.h>
#include <api/output_descriptor.h>
#include <api/platform_info.h>

// Stop conditions
#include <api/stop_condition/configuration_duration.h>
#include <api/stop_condition/configuration_count.h>
#include <api/stop_condition/configuration_fraction.h>
#include <api/stop_condition/tuning_duration.h>

// Reference class interface
#include <api/reference_class.h>

// Tuning manipulator interface
#include <api/tuning_manipulator.h>

// Support for 16-bit floating point data type
#include <half.hpp>

/** @namespace ktt
  * All classes, methods and type aliases related to KTT framework are located inside ktt namespace.
  */
namespace ktt
{

using half_float::half;
class TunerCore;

/** @class Tuner
  * Class which serves as the main part of public API of KTT framework.
  */
class KTT_API Tuner
{
public:
    /** @fn explicit Tuner(const PlatformIndex platform, const DeviceIndex device)
      * Constructor, which creates new tuner object for specified platform and device. Tuner uses OpenCL as compute API, all commands are
      * submitted to a single compute queue. Indices for available platforms and devices can be retrieved by calling printComputeAPIInfo() method.
      * @param platform Index for platform used by created tuner.
      * @param device Index for device used by created tuner.
      */
    explicit Tuner(const PlatformIndex platform, const DeviceIndex device);

    /** @fn explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI)
      * Constructor, which creates new tuner object for specified platform, device and compute API. All commands are submitted to a single
      * compute queue. Indices for available platforms and devices can be retrieved by calling printComputeAPIInfo() method. If specified compute API
      * is CUDA, platform index is ignored.
      * @param platform Index for platform used by created tuner.
      * @param device Index for device used by created tuner.
      * @param computeAPI Compute API used by created tuner.
      */
    explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI);

    /** @fn explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI, const uint32_t computeQueueCount)
      * Constructor, which creates new tuner object for specified platform, device and compute API. Several compute queues are created, based
      * on specified count. Commands to different queues can be submitted by utilizing TuningManipulator. Indices for available platforms and devices
      * can be retrieved by calling printComputeAPIInfo() method. If specified compute API is CUDA, platform index is ignored.
      * @param platform Index for platform used by created tuner.
      * @param device Index for device used by created tuner.
      * @param computeAPI Compute API used by created tuner.
      * @param computeQueueCount Number of compute queues created inside the tuner. Has to be greater than zero.
      */
    explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI, const uint32_t computeQueueCount);

    /** @fn ~Tuner()
      * Tuner destructor.
      */
    ~Tuner();

    /** @fn KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
      * const DimensionVector& localSize)
      * Adds new kernel to tuner from source code inside string. Requires specification of kernel name and default global and local thread sizes.
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
      * Adds new kernel to tuner from source code inside file. Requires specification of kernel name and default global and local thread sizes.
      * @param filePath Path to file with kernel source code written in corresponding compute API language.
      * @param kernelName Name of kernel function inside kernel source code.
      * @param globalSize Dimensions for base kernel global size (eg. grid size in CUDA).
      * @param localSize Dimensions for base kernel local size (eg. block size in CUDA).
      * @return Id assigned to kernel by tuner. The id can be used in other API methods.
      */
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    /** @fn void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
      * Sets kernel arguments for specified kernel by providing corresponding argument ids.
      * @param id Id of kernel for which the arguments will be set.
      * @param argumentIds Ids of arguments to be used by specified kernel. Order of ids must match the order of kernel arguments specified in kernel
      * function. Argument ids for single kernel must be unique.
      */
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);

    /** @fn void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues)
      * Adds new integer parameter for specified kernel, providing parameter name and list of allowed values. When the corresponding kernel
      * is launched, parameters will be added to kernel source code as preprocessor definitions. During the tuning process, tuner will generate
      * configurations for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter will be added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues Vector of allowed values for the parameter.
      */
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues);

    /** @fn void addParameterDouble(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues)
      * Adds new floating-point parameter for specified kernel, providing parameter name and list of allowed values. When the corresponding
      * kernel is launched, parameters will be added to kernel source code as preprocessor definitions. During the tuning process, tuner will
      * generate configurations for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter will be added.
      * @param parameterName Name of a parameter. Parameter names for single kernel must be unique.
      * @param parameterValues Vector of allowed values for the parameter.
      */
    void addParameterDouble(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues);

    /** @fn void addParameterPack(const KernelId id, const std::string& packName, const std::vector<std::string> parameterNames)
      * Adds a pack containing specified kernel parameters. When parameter packs are used, tuning configurations are generated progressively for each
      * pack. Once best configuration is found for a specific pack, next pack is then processed. This method is useful when kernels contain groups of
      * parameters that can be tuned independently. The total number of generated configurations that need to be tested is reduced.
      * @param id Id of kernel for which the parameter pack will be added.
      * @param packName Name of a parameter pack. Parameter pack names for a single kernel must be unique.
      * @param parameterNames Names of parameters which will be added to the parameter pack.
      */
    void addParameterPack(const KernelId id, const std::string& packName, const std::vector<std::string>& parameterNames);

    /** @fn void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
      * const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
      * Sets a thread modifier function for specified kernel. This function receives thread size in specified kernel dimension and values of
      * specified kernel parameters as input and returns modified thread size based on these values. The modifiers are useful in case when
      * kernel parameters affect the number of required kernel threads.
      * @param id Id of kernel for which the modifier will be set.
      * @param modifierType Type of the thread modifier. See ::ModifierType for more information.
      * @param modifierDimension Dimension which will be affected by the modifier. See ::ModifierDimension for more information.
      * @param parameterNames Names of kernel parameters whose values will be passed into the modifier function. The order of parameter names
      * corresponds to the order of parameter values inside the modifier function vector argument.
      * @param modifierFunction Function which receives thread size in specified kernel dimension and values of specified kernel parameters as input
      * and returns modified thread size based on these values.
      */
    void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);

    /** @fn void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
      * const std::string& parameterName, const ModifierAction modifierAction)
      * Sets a thread modifier for specified kernel. This is a simplified version of the thread modifier method which only supports one parameter
      * and limited number of actions, but is easier to use.
      * @param id Id of kernel for which the modifier will be set.
      * @param modifierType Type of the thread modifier. See ::ModifierType for more information.
      * @param modifierDimension Dimension which will be affected by the thread modifier. See ::ModifierDimension for more information.
      * @param parameterName Name of a kernel parameter whose value will be utilized by the thread modifier.
      * @param modifierAction Action of the thread modifier. See ::ModifierAction for more information.
      */
    void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
        const std::string& parameterName, const ModifierAction modifierAction);

    /** @fn void setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
      * const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
      * Sets a local memory modifier function for specified kernel. This function receives size of specified local memory argument and values of
      * specified kernel parameters as input and returns modified argument size based on these values. The modifiers are useful in case when kernel
      * parameters affect the size of local memory arguments.
      * @param id Id of kernel for which the modifier will be set.
      * @param argumentId Id of local memory kernel argument which will be affected by the modifier.
      * @param parameterNames Names of kernel parameters whose values will be passed into the modifier function. The order of parameter names
      * corresponds to the order of parameter values inside the modifier function vector argument.
      * @param modifierFunction Function which receives size of specified local memory argument and values of specified kernel parameters as input
      * and returns modified argument size based on these values.
      */
    void setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);

    /** @fn void addConstraint(const KernelId id, const std::vector<std::string>& parameterNames,
      * const std::function<bool(const std::vector<size_t>&)>& constraintFunction)
      * Adds new constraint for specified kernel. Constraints are used to prevent generating of invalid configurations (eg. conflicting parameter
      * values).
      * @param id Id of kernel for which the constraint will be added.
      * @param parameterNames Names of kernel parameters which will be affected by the constraint function. The order of parameter names corresponds
      * to the order of parameter values inside the constraint function vector argument.
      * @param constraintFunction Function which returns true if provided combination of parameter values is valid. Returns false otherwise.
      */
    void addConstraint(const KernelId id, const std::vector<std::string>& parameterNames,
        const std::function<bool(const std::vector<size_t>&)>& constraintFunction);

    /** @fn void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
      * Sets tuning manipulator for specified kernel. Tuning manipulator enables customization of kernel execution. This is useful in several cases,
      * eg. running part of the computation in C++ code, utilizing iterative kernel launches or composite kernels. See TuningManipulator for more
      * information.
      * @param id Id of kernel for which the tuning manipulator will be set.
      * @param manipulator Tuning manipulator for specified kernel.
      */
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    /** @fn void setTuningManipulatorSynchronization(const KernelId id, const bool flag)
      * Controls whether framework performs implicit device synchronization after launchComputation() method inside tuning manipulator has finished.
      * Performing the synchronization is necessary for obtaining correct computation duration. Turning it off may increase performance during
      * regular computation after the tuning has concluded. Synchronization is turned on by default.
      * @param id Id of kernel for which the tuning manipulator synchronization flag will be set.
      * @param flag If true, tuning manipulator synchronization will be turned on for specified kernel or kernel composition. It will be turned off
      * otherwise.
      */
    void setTuningManipulatorSynchronization(const KernelId id, const bool flag);

    /** @fn KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
      * std::unique_ptr<TuningManipulator> manipulator)
      * Creates a kernel composition using specified kernels. The following methods can be used with kernel compositions and will call
      * the corresponding method for all kernels inside the composition: setKernelArguments(), addParameter() (both versions), addConstraint().
      * 
      * Kernel compositions do not inherit any parameters or constraints from the original kernels. Setting kernel arguments and adding parameters
      * or constraints to kernels inside given composition will not affect the original kernels or other compositions. Tuning manipulator is required
      * in order to launch kernel composition with the tuner. See TuningManipulator for more information.
      * @param compositionName Name of kernel composition. The name is used during output printing.
      * @param kernelIds Ids of kernels which will be included in the composition.
      * @param manipulator Tuning manipulator for the composition.
      * @return Id assigned to kernel composition by tuner. The id can be used in other API methods.
      */
    KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
        std::unique_ptr<TuningManipulator> manipulator);

    /** @fn void setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
      * const ModifierDimension modifierDimension, const std::vector<std::string>& parameterNames,
      * const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
      * Calls setThreadModifier() method for a single kernel inside specified kernel composition. Does not affect standalone kernels or other
      * compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the modifier will be set.
      * @param modifierType Type of the thread modifier. See ::ModifierType for more information.
      * @param modifierDimension Dimension which will be affected by the thread modifier. See ::ModifierDimension for more information.
      * @param parameterNames Names of kernel parameters whose values will be passed into the modifier function. The order of parameter names
      * corresponds to the order of parameter values inside the modifier function vector argument.
      * @param modifierFunction Function which receives thread size in specified kernel dimension and values of specified kernel parameters as input
      * and returns modified thread size based on these values.
      */
    void setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
        const ModifierDimension modifierDimension, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);

    /** @fn void setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
      * const ModifierDimension modifierDimension, const std::string& parameterName, const ModifierAction modifierAction)
      * Calls simplified version of setThreadModifier() method for a single kernel inside specified kernel composition. Does not affect standalone
      * kernels or other compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the modifier will be set.
      * @param modifierType Type of the thread modifier. See ::ModifierType for more information.
      * @param modifierDimension Dimension which will be affected by the thread modifier. See ::ModifierDimension for more information.
      * @param parameterName Name of a kernel parameter whose value will be utilized by the thread modifier.
      * @param modifierAction Action of the thread modifier. See ::ModifierAction for more information.
      */
    void setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
        const ModifierDimension modifierDimension, const std::string& parameterName, const ModifierAction modifierAction);

    /** @fn void setCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const ArgumentId argumentId,
      * const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction)
      * Calls setLocalMemoryModifier() method for a single kernel inside specified kernel composition. Does not affect standalone kernels or other
      * compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the modifier will be set.
      * @param argumentId Id of local memory kernel argument which will be affected by the modifier.
      * @param parameterNames Names of kernel parameters whose values will be passed into the modifier function. The order of parameter names
      * corresponds to the order of parameter values inside the modifier function vector argument.
      * @param modifierFunction Function which receives size of specified local memory argument and values of specified kernel parameters as input
      * and returns modified argument size based on these values.
      */
    void setCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const ArgumentId argumentId,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);

    /** @fn void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds)
      * Calls setKernelArguments() method for a single kernel inside specified kernel composition. Does not affect standalone kernels or other
      * compositions.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the arguments will be set.
      * @param argumentIds Ids of arguments to be used by specified kernel inside the composition. Order of ids must match the order of kernel
      * arguments specified in kernel function. Argument ids for single kernel must be unique.
      */
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    /** @fn template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType accessType)
      * Adds new vector argument to tuner. Makes copy of argument data, so the source data vector remains unaffected by tuner operations.
      * Argument data will be accessed from device memory during its usage by compute API.
      * @param data Argument data provided in std::vector. Provided data type must be trivially copyable. Bool data type is currently not supported.
      * @param accessType Access type of argument specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentVector(const std::vector<T>& data, const ArgumentAccessType accessType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, ArgumentMemoryLocation::Device, accessType, ArgumentUploadType::Vector);
    }

    /** @fn template <typename T> ArgumentId addArgumentVector(std::vector<T>& data, const ArgumentAccessType accessType,
      * const ArgumentMemoryLocation memoryLocation, const bool copyData)
      * Adds new vector argument to tuner. Allows choice for argument memory location and whether argument data is copied to tuner.
      * @param data Argument data provided in std::vector. Provided data type must be trivially copyable. Bool data type is currently not supported.
      * @param accessType Access type of argument specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @param memoryLocation Memory location of argument specifies whether argument will be accessed from device or host memory during its usage
      * by compute API. See ::ArgumentMemoryLocation for more information.
      * @param copyData Flag which specifies whether the argument is copied inside tuner. If set to false, tuner will store reference of source data
      * vector and will access it directly during kernel launch operations. This results in lower memory overhead, but relies on a user to keep data
      * in source vector valid.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentVector(std::vector<T>& data, const ArgumentAccessType accessType,
        const ArgumentMemoryLocation memoryLocation, const bool copyData)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), sizeof(T), dataType, memoryLocation, accessType, copyData);
    }

    /** @fn template <typename T> ArgumentId addArgumentScalar(const T& data)
      * Adds new scalar argument to tuner. All scalar arguments are read-only.
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
      * Adds new local memory (shared memory in CUDA) argument to tuner. All local memory arguments are read-only and cannot be initialized
      * from host memory. In case of CUDA API usage, local memory arguments cannot be directly set as kernel function arguments. Setting a local
      * memory argument to kernel in CUDA means that corresponding amount of memory will be allocated for kernel to use. In that case, all local
      * memory argument ids should be specified at the end of the vector when calling setKernelArguments() method.
      * @param localMemoryElementsCount Specifies how many elements of provided data type the argument contains.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T> ArgumentId addArgumentLocal(const size_t localMemoryElementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(localMemoryElementsCount, sizeof(T), dataType);
    }

    /** @fn void persistArgument(const ArgumentId id, const bool flag)
      * Controls whether specified vector argument is persisted inside a compute API buffer or not. Persisted arguments remain inside buffers even
      * after the execution of kernel utilizing these arguments has finished. Persistence of kernel arguments is switched off by default. Persistent
      * arguments can be useful during online tuning and regular kernel running, when kernel output is computed over multiple kernel launches in
      * different configurations. Note that persistent arguments are never utilized by reference kernels.
      * @param id Id of a vector argument.
      * @param flag Specifies whether argument should be persisted or not. If true, specified vector argument is immediately persisted. If false,
      * compute API buffer for specified argument is immediately destroyed.
      */
    void persistArgument(const ArgumentId id, const bool flag);

    /** @fn void downloadPersistentArgument(const OutputDescriptor& output) const
      * Downloads specified persistent argument from compute API buffer into specified memory location.
      * @param output User-provided memory location for persistent kernel argument which should be retrieved. See OutputDescriptor for more information.
      */
    void downloadPersistentArgument(const OutputDescriptor& output) const;

    /** @fn std::vector<ComputationResult> tuneKernel(const KernelId id)
      * Starts the tuning process for specified kernel. Creates configuration space based on combinations of provided kernel parameters and
      * constraints. The configurations will be launched in order that depends on specified ::SearchMethod. Tuning will end when all configurations
      * are explored.
      * @param id Id of kernel for which the tuning will start.
      * @return Vector of objects containing information about computation using tested kernel configurations. See ComputationResult for more
      * information.
      */
    std::vector<ComputationResult> tuneKernel(const KernelId id);

    /** @fn std::vector<ComputationResult> tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
      * Starts the tuning process for specified kernel. Creates configuration space based on combinations of provided kernel parameters and
      * constraints. The configurations will be launched in order that depends on specified ::SearchMethod. Tuning will end either when all
      * configurations are explored or when specified stop condition is met.
      * @param id Id of kernel for which the tuning will start.
      * @param stopCondition Stop condition which decides whether to continue the tuning process. See StopCondition for more information.
      * @return Vector of objects containing information about computation using tested kernel configurations. See ComputationResult for more
      * information.
      */
    std::vector<ComputationResult> tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition);

    /** @fn std::vector<ComputationResult> dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations = 0)
      * Starts the simulated tuning process for specified kernel (kernel is not tuned, execution times are read from CSV). Creates configuration
      * space based on combinations of provided kernel parameters and constraints. The configurations will be launched in order that depends on
      * specified ::SearchMethod. This method can be used to test behaviour and performance of newly implemented search methods. Note that no checks
      * are performed whether the tuning data relates to kernel, tuning parameters or hardware. It is up to user to ensure that dryTuneKernel() reads
      * a valid file.
      * @param id Id of kernel for which the tuning begins.
      * @param filePath Path to CSV file with tuning parameters.
      * @param iterations Number of iterations performed, 0 = scan whole tuning space.
      * @return Vector of objects containing information about computation using tested kernel configurations. See ComputationResult for more
      * information.
      */
    std::vector<ComputationResult> dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations = 0);

    /** @fn ComputationResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output)
      * Performs one step of the tuning process for specified kernel. When this method is called inside tuner for the first time, it creates
      * configuration space based on combinations of provided kernel parameters and constraints. Each time this method is called, it launches single
      * kernel configuration. If all configurations were already tested, runs kernel using the best configuration. Output data can be retrieved
      * by providing output descriptors. Always performs recomputation of reference output.
      * @param id Id of kernel for which the tuning by step will start.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See OutputDescriptor for more information.
      * @return Object containing information about computation using current kernel configuration. See ComputationResult for more information.
      */
    ComputationResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output);

    /** @fn ComputationResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference)
      * Performs one step of the tuning process for specified kernel. When this method is called inside tuner for the first time, it creates
      * configuration space based on combinations of provided kernel parameters and constraints. Each time this method is called, it launches single
      * kernel configuration. If all configurations were already tested, runs kernel using the best configuration. Output data can be retrieved
      * by providing output descriptors. Allows control over recomputation of reference output.
      * @param id Id of kernel for which the tuning by step will start.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See OutputDescriptor for more information.
      * @param recomputeReference Flag which controls whether recomputation of reference output should be performed or not. Useful if kernel data
      * between individual method invocations sometimes change.
      * @return Object containing information about computation using current kernel configuration. See ComputationResult for more information.
      */
    ComputationResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference);

    /** @fn ComputationResult runKernel(const KernelId id, const std::vector<ParameterPair>& configuration,
      * const std::vector<OutputDescriptor>& output)
      * Runs specified kernel using provided configuration.
      * @param id Id of kernel which will be run.
      * @param configuration Configuration under which the kernel will be launched. See ParameterPair for more information.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See OutputDescriptor for more information.
      * @return Object containing information about computation using specified kernel configuration. See ComputationResult for more information.
      */
    ComputationResult runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output);

    /** @fn void clearKernelData(const KernelId id, const bool clearConfigurations)
      * Resets tuning process and clears tuning results for specified kernel.
      * @param id Id of kernel whose data will be cleared.
      * @param clearConfigurations If true, generated kernel configurations will be cleared as well. Otherwise, they will remain inside tuner.
      */
    void clearKernelData(const KernelId id, const bool clearConfigurations);

    /** @fn void setKernelProfiling(const bool flag)
      * Toggles profiling of kernel runs inside the tuner. Profiled kernel runs generate profiling counters which can be used by searchers and stop
      * conditions for more accurate performance measurement. Profiling counters can also be retrieved through API and printed into CSV file with
      * tuning results. Note that enabling profiling will result in longer tuning times because profiled kernels have to be launched multiple times
      * with the same configuration in order to collect all profiling counters. Asynchronous kernel launches are currently not supported when kernel
      * profiling is enabled. Kernel profiling is disabled by default.
      * @param flag If true, kernel profiling is enabled. It is disabled otherwise.
      */
    void setKernelProfiling(const bool flag);

    /** @fn void setCompositionKernelProfiling(const KernelId compositionId, const KernelId kernelId, const bool flag)
      * Toggles profiling of a specific kernel inside kernel composition. This is useful if only some kernels inside the composition need to be
      * profiled. By default, profiling is enabled for all kernels inside newly added kernel compositions. Note that this method has no effect if
      * kernel profiling is completely disabled. See setKernelProfiling() method for more information.
      * @param compositionId Id of composition which includes the specified kernel.
      * @param kernelId Id of kernel inside the composition for which the profiling flag will be set.
      * @param flag If true, kernel profiling is enabled for specified kernel inside the composition. It is disabled otherwise.
      */
    void setCompositionKernelProfiling(const KernelId compositionId, const KernelId kernelId, const bool flag);

    /** @fn void setKernelProfilingCounters(const std::vector<std::string>& counterNames)
      * Specifies profiling counters that will be collected during kernel profiling. Note that not all profiling counters are available on all
      * devices.
      *
      * For the list of CUDA CUPTI profiling counters, see: https://docs.nvidia.com/cuda/cupti/index.html#metrics-reference
      * @param counterNames Names of counters that will be collected during kernel profiling.
      */
    void setKernelProfilingCounters(const std::vector<std::string>& counterNames);

    /** @fn void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
      * Specifies search method which will be used during kernel tuning. Number of required search arguments depends on the search method.
      * Default search method is full search.
      * @param method Search method which will be used during kernel tuning. See SearchMethod for more information.
      * @param arguments Arguments necessary for specified search method to work. Following arguments are required for corresponding search method,
      * the order of arguments is important:
      * - FullSearch - none
      * - RandomSearch - none
      * - Annealing - maximum temperature
      * - MCMC - none
      */
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);

    /** @fn void setPrintingTimeUnit(const TimeUnit unit)
      * Sets time unit used for printing of results. Default time unit is milliseconds. 
      * @param unit Time unit which will be used for printing of results. See ::TimeUnit for more information.
      */
    void setPrintingTimeUnit(const TimeUnit unit);

    /** @fn void setInvalidResultPrinting(const bool flag)
      * Toggles printing of results from failed kernel runs. Invalid results will be separated from valid results during printing.
      * Printing of invalid results is disabled by default.
      * @param flag If true, printing of invalid results will be enabled. It will be disabled otherwise.
      */
    void setInvalidResultPrinting(const bool flag);

    /** @fn void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const
      * Prints tuning results for specified kernel to specified output stream. Valid results will be printed only if method tuneKernel() or
      * tuneKernelByStep() was already called for corresponding kernel.
      * @param id Id of kernel for which the results will be printed.
      * @param outputTarget Location where the results will be printed.
      * @param format Format in which the results will be printed. See ::PrintFormat for more information.
      */
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const;

    /** @fn void printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const
      * Prints tuning results for specified kernel to specified file. Valid results will be printed only if method tuneKernel() or
      * tuneKernelByStep() was already called for corresponding kernel.
      * @param id Id of kernel for which the results will be printed.
      * @param filePath Path to file where the results will be printed.
      * @param format Format in which the results are printed. See ::PrintFormat for more information.
      */
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const;

    /** @fn ComputationResult getBestComputationResult(const KernelId id) const
      * Returns the best computation result found for specified kernel. Valid result will be returned only if one of the kernel tuning methods was
      * already called for corresponding kernel.
      * @param id Id of kernel for which the best result will be returned.
      * @return Object containing information about best computation result for specified kernel. See ComputationResult for more information.
      */
    ComputationResult getBestComputationResult(const KernelId id) const;

    /** @fn std::string getKernelSource(const KernelId id, const std::vector<ParameterPair>& configuration) const
      * Returns kernel source with preprocessor definitions for specified kernel based on provided configuration.
      * @param id Id of kernel for which the source is returned.
      * @param configuration Kernel configuration for which the source will be generated. See ParameterPair for more information.
      * @return Kernel source with preprocessor definitions for specified kernel based on provided configuration.
      */
    std::string getKernelSource(const KernelId id, const std::vector<ParameterPair>& configuration) const;

    /** @fn void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
      * const std::vector<ArgumentId>& validatedArgumentIds)
      * Sets reference kernel for specified kernel. Reference kernel output will be compared to tuned kernel output in order to ensure
      * correctness of computation. Reference kernel uses only single configuration and cannot be composite.
      * @param id Id of kernel for which reference kernel will be set.
      * @param referenceId Id of reference kernel. This can be the same as validated kernel. This can be useful if the kernel has a configuration
      * which is known to produce correct results.
      * @param referenceConfiguration Configuration under which the reference kernel will be launched to produce reference output.
      * @param validatedArgumentIds Ids of kernel arguments which will be validated. The validated arguments must be vector arguments and cannot be
      * read-only.
      */
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);

    /** @fn void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
      * const std::vector<ArgumentId>& validatedArgumentIds)
      * Sets reference class for specified kernel. Reference class output will be compared to tuned kernel output in order to ensure
      * correctness of computation.
      * @param id Id of kernel for which reference class will be set.
      * @param referenceClass Reference class which produces reference output for specified kernel. See ReferenceClass for more information.
      * @param validatedArgumentIds Ids of kernel arguments which will be validated. The validated arguments must be vector arguments and cannot be
      * read-only.
      */
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);

    /** @fn void setValidationMode(const ValidationMode mode)
      * Sets modes under which kernel output validation is enabled. By default, output validation is enabled only during kernel tuning.
      * @param mode Bitfield of modes under which kernel output validation is enabled. See ::ValidationMode for more information.
      */
    void setValidationMode(const ValidationMode mode);

    /** @fn void setValidationMethod(const ValidationMethod method, const double toleranceThreshold)
      * Sets validation method and tolerance threshold for floating-point argument validation. Default validation method is side by side comparison.
      * Default tolerance threshold is 1e-4.
      * @param method Validation method which will be used for floating-point argument validation. See ::ValidationMethod for more information.
      * @param toleranceThreshold Output validation threshold. If difference between tuned kernel output and reference output is within tolerance
      * threshold, the tuned kernel output will be considered correct.
      */
    void setValidationMethod(const ValidationMethod method, const double toleranceThreshold);

    /** @fn void setValidationRange(const ArgumentId id, const size_t range)
      * Sets validation range for specified argument to specified validation range. Only elements within validation range, starting with the first
      * element, will be validated. All elements are validated by default.
      * @param id Id of argument for which the validation range will be set.
      * @param range Range inside which the argument elements will be validated, starting from the first element.
      */
    void setValidationRange(const ArgumentId id, const size_t range);

    /** @fn void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
      * Sets argument comparator for specified kernel argument. Arguments with custom data type cannot be compared using built-in comparison
      * operators and require user to provide a comparator. Comparator can also be optionally added for arguments with built-in data types.
      * @param id Id of argument for which the comparator will be set.
      * @param comparator Function which receives two elements with data type matching the data type of specified kernel argument and returns true
      * if the elements are equal. Returns false otherwise.
      */
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);

    /** @fn void setCompilerOptions(const std::string& options)
      * Sets compute API compiler options to specified options. There are no default options for OpenCL back-end. Default option for CUDA
      * back-end is "--gpu-architecture=compute_30".
      * 
      * For list of OpenCL compiler options, see: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clBuildProgram.html
      * For list of CUDA compiler options, see: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options
      * @param options Compute API compiler options. If multiple options are used, they need to be separated by a single space character.
      */
    void setCompilerOptions(const std::string& options);

    /** @fn void setKernelCacheCapacity(const size_t capacity)
      * Sets capacity of kernel cache inside the tuner. The cache contains recently compiled kernels which are prepared to be launched immidiately,
      * eliminating compilation overhead. Using the cache can significantly improve tuner performance during online tuning or iterative kernel
      * running with TuningManipulator. Default cache size is 10.
      * @param capacity Controls kernel cache capacity. If zero, kernel cache is completely disabled.
      */
    void setKernelCacheCapacity(const size_t capacity);

    /** @fn void printComputeAPIInfo(std::ostream& outputTarget) const
      * Prints basic information about available platforms and devices to specified output stream. Also prints indices assigned to them
      * by KTT framework.
      * @param outputTarget Location where the information will be printed.
      */
    void printComputeAPIInfo(std::ostream& outputTarget) const;

    /** @fn std::vector<PlatformInfo> getPlatformInfo() const
      * Retrieves detailed information about all available platforms (eg. platform name, vendor). See PlatformInfo for more information.
      * @return Information about all available platforms.
      */
    std::vector<PlatformInfo> getPlatformInfo() const;

    /** @fn std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const
      * Retrieves detailed information about all available devices (eg. device name, memory capacity) on specified platform. See DeviceInfo
      * for more information.
      * @param platform Index of platform for which the device information will be retrieved.
      * @return Information about all available devices on specified platform.
      */
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const;

    /** @fn DeviceInfo getCurrentDeviceInfo() const
      * Retrieves detailed information about device (eg. device name, memory capacity) used by the tuner. See DeviceInfo for more information.
      * @return Information about device used by the tuner.
      */
    DeviceInfo getCurrentDeviceInfo() const;

    /** @fn void setAutomaticGlobalSizeCorrection(const bool flag)
      * Toggles automatic correction for global size, which ensures that global size in each dimension is always a multiple of local size in
      * corresponding dimension. Performs a roundup to the nearest higher multiple. Automatic global size correction is disabled by default.
      * @param flag If true, automatic global size correction will be enabled. It will be disabled otherwise.
      */
    void setAutomaticGlobalSizeCorrection(const bool flag);

    /** @fn void setGlobalSizeType(const GlobalSizeType type)
      * Sets global size specification type to specified compute API style. In OpenCL, NDrange size is specified as number of work-items in
      * a work-group multiplied by number of work-groups. In CUDA, grid size is specified as number of blocks. This method makes it possible to use
      * OpenCL style in CUDA and vice versa. Default global size type is the one corresponding to compute API of the tuner.
      * @param type Global size type which will be set for tuner. See ::GlobalSizeType for more information.
      */
    void setGlobalSizeType(const GlobalSizeType type);

     /** @fn static void setLoggingLevel(const LoggingLevel level)
      * Sets logging level for tuner. Default logging level is info.
      * @param level Logging level which will be used by tuner. See ::LoggingLevel for more information.
      */
    static void setLoggingLevel(const LoggingLevel level);

    /** @fn static void setLoggingTarget(std::ostream& outputTarget)
      * Sets the target for info messages logging to specified output stream. Default logging target is `std::clog`.
      * @param outputTarget Location where tuner info messages will be printed.
      */
    static void setLoggingTarget(std::ostream& outputTarget);

    /** @fn static void setLoggingTarget(const std::string& filePath)
      * Sets the target for info messages logging to specified file. Default logging target is `std::clog`.
      * @param filePath Path to file where tuner info messages will printed.
      */
    static void setLoggingTarget(const std::string& filePath);

private:
    // Pointer to implementation class
    std::unique_ptr<TunerCore> tunerCore;

    // Helper methods
    ArgumentId addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const bool copyData);
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType);
    ArgumentId addArgument(const size_t localMemoryElementsCount, const size_t elementSizeInBytes, const ArgumentDataType dataType);

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
