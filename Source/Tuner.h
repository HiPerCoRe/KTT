/** @file Tuner.h
  * Main part of public API of KTT framework.
  */
#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Compatibility for multiple platforms
#include <KttPlatform.h>

// Data types and enums
#include <ComputeEngine/ComputeApi.h>
#include <ComputeEngine/GlobalSizeType.h>
#include <Kernel/ModifierAction.h>
#include <Kernel/ModifierDimension.h>
#include <Kernel/ModifierType.h>
#include <KernelArgument/ArgumentAccessType.h>
#include <KernelArgument/ArgumentDataType.h>
#include <KernelArgument/ArgumentManagementType.h>
#include <KernelArgument/ArgumentMemoryLocation.h>
#include <KernelArgument/ArgumentMemoryType.h>
#include <KernelRunner/ValidationMethod.h>
#include <KernelRunner/ValidationMode.h>
#include <Output/TimeConfiguration/TimeUnit.h>
#include <Output/OutputFormat.h>
#include <Utility/Logger/LoggingLevel.h>
#include <KttTypes.h>

// Data holders
#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Info/DeviceInfo.h>
#include <Api/Info/PlatformInfo.h>
#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>

// Tuner customization
#include <Api/Searcher/Searcher.h>
#include <Api/StopCondition/StopCondition.h>
#include <Api/ComputeApiInitializer.h>

// Half floating-point data type support
#include <Utility/External/half.hpp>

/** @namespace ktt
  * All classes, methods and type aliases related to KTT framework are located inside ktt namespace.
  */
namespace ktt
{

class TunerCore;

/** @class Tuner
  * Class which serves as the main part of public API of KTT framework.
  */
class KTT_API Tuner
{
public:
    /** @fn explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api)
      * Creates tuner for the specified platform, device and compute API. All compute commands are submitted to a single queue.
      * Indices for available platforms and devices can be retrieved by using GetPlatformInfo() and GetDeviceInfo() methods. If
      * the specified compute API is CUDA or Vulkan, platform index is ignored.
      * @param platform Index for platform used by the tuner.
      * @param device Index for device used by the tuner.
      * @param api Compute API used by the tuner.
      */
    explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api);

    /** @fn explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api,
      * const uint32_t computeQueueCount)
      * Creates tuner for the specified platform, device and compute API. Multiple compute queues can be created, based on the
      * specified count. Compute commands to different queues can be submitted by utilizing KernelLauncher and ComputeInterface.
      * Indices for available platforms and devices can be retrieved by using GetPlatformInfo() and GetDeviceInfo() methods. If
      * the specified compute API is CUDA or Vulkan, platform index is ignored.
      * @param platform Index for platform used by the tuner.
      * @param device Index for device used by the tuner.
      * @param api Compute API used by the tuner.
      * @param computeQueueCount Number of compute queues created inside the tuner. Has to be greater than zero.
      */
    explicit Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t computeQueueCount);

    /** @fn explicit Tuner(const ComputeApi api, const ComputeApiInitializer& initializer)
      * Creates tuner for the specified compute API using custom initializer. The initializer contains user-provided compute device
      * context and queues.
      * @param api Compute API used by the tuner.
      * @param initializer Custom compute API initializer. See ComputeApiInitializer for more information.
      */
    explicit Tuner(const ComputeApi api, const ComputeApiInitializer& initializer);

    /** @fn ~Tuner()
      * Tuner destructor.
      */
    ~Tuner();

    /** @fn KernelDefinitionId AddKernelDefinition(const std::string& name, const std::string& source,
      * const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {})
      * Adds new kernel definition to the tuner. Requires specification of a kernel name, its source code and default global and
      * local thread sizes.
      * @param name Name of a kernel function inside kernel source code. The name must be unique.
      * @param source Kernel source code written in the corresponding compute API language.
      * @param globalSize Dimensions for base kernel global size (e.g., grid size in CUDA, NDRange size in OpenCL).
      * @param localSize Dimensions for base kernel local size (e.g., block size in CUDA, work-group size in OpenCL).
      * @param typeNames Names of types which will be used to instantiate kernel template. Only supported in CUDA kernels.
      * @return Id assigned to kernel definition by the tuner. The id can be used in other API methods.
      */
    KernelDefinitionId AddKernelDefinition(const std::string& name, const std::string& source, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});

    /** @fn KernelDefinitionId AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
      * const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {})
      * Adds new kernel definition to the tuner. Requires specification of a kernel name, file path to its source code and default
      * global and local thread sizes.
      * @param name Name of a kernel function inside kernel source code. The name must be unique.
      * @param filePath Path to file with kernel source code written in the corresponding compute API language.
      * @param globalSize Dimensions for base kernel global size (e.g., grid size in CUDA, NDRange size in OpenCL).
      * @param localSize Dimensions for base kernel local size (e.g., block size in CUDA, work-group size in OpenCL).
      * @param typeNames Names of types which will be used to instantiate kernel template. Only supported in CUDA kernels.
      * @return Id assigned to kernel definition by the tuner. The id can be used in other API methods.
      */
    KernelDefinitionId AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
        const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});

    /** @fn void RemoveKernelDefinition(const KernelDefinitionId id)
      * Removes kernel definition with the specified id from the tuner. Note that definition can only be removed if it is not
      * associated with any kernel.
      * @param id Id of the kernel definition which will be removed.
      */
    void RemoveKernelDefinition(const KernelDefinitionId id);

    /** @fn void SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
      * Sets arguments for the specified kernel definition.
      * @param id Id of a kernel definition for which the arguments will be set.
      * @param argumentIds Ids of arguments to be used by the specified definition. The order of ids must match the order of
      * kernel arguments inside kernel function. The provided ids must be unique.
      */
    void SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds);

    /** @fn KernelId CreateSimpleKernel(const KernelDefinitionId definitionId)
      * Creates simple kernel from the specified definition.
      * @param name Kernel name used during logging and output operations. The name must be unique.
      * @param definitionId Id of kernel definition which will be utilized by the kernel.
      * @return Id assigned to kernel by the tuner. The id can be used in other API methods.
      */
    KernelId CreateSimpleKernel(const std::string& name, const KernelDefinitionId definitionId);

    /** @fn KernelId CreateCompositeKernel(const std::vector<KernelDefinitionId>& definitionIds, KernelLauncher launcher = nullptr)
      * Creates composite kernel from the specified definitions. Note that kernel launcher is required in order to launch kernels
      * with multiple definitions.
      * @param name Kernel name used during logging and output operations. The name must be unique.
      * @param definitionIds Ids of kernel definitions which will be utilized by the kernel.
      * @param launcher Launcher for the kernel. It can be defined either during kernel creation or later with SetLauncher() method.
      * @return Id assigned to kernel by the tuner. The id can be used in other API methods.
      */
    KernelId CreateCompositeKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds,
        KernelLauncher launcher = nullptr);

    /** @fn void RemoveKernel(const KernelId id)
      * Removes kernel with the specified id from the tuner. If the kernel is used as a reference kernel, the corresponding kernel
      * argument output validation will be disabled.
      * @param id Id of the kernel which will be removed.
      */
    void RemoveKernel(const KernelId id);

    /** @fn void SetLauncher(const KernelId id, KernelLauncher launcher)
      * Specifies kernel launcher for a kernel. Kernel launcher enables customization of kernel execution. This is useful in
      * multiple cases. E.g., running part of the computation in C++ code, utilizing iterative kernel launches or kernels with
      * multiple definitions.
      * @param id Id of kernel for which launcher will be set.
      * @param launcher Launcher for the specified kernel. See ComputeInterface for further information.
      */
    void SetLauncher(const KernelId id, KernelLauncher launcher);

    /** @fn void AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values,
      * const std::string& group = "")
      * Adds new integer parameter for the specified kernel, providing parameter name and list of allowed values. Parameters will
      * be added to the kernel source code as preprocessor definitions. During the tuning process, tuner will generate configurations
      * for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter will be added.
      * @param name Name of a parameter. Parameter names for a single kernel must be unique.
      * @param values Allowed values for the parameter.
      * @param group Optional group inside which the parameter will be added. Tuning configurations are generated separately for each
      * group. This is useful when kernels contain groups of parameters that can be tuned independently. In this way, the total number
      * of generated configurations can be significantly reduced.
      */
    void AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values, const std::string& group = "");

    /** @fn void AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values,
      * const std::string& group = "")
      * Adds new floating-point parameter for the specified kernel, providing parameter name and list of allowed values. Parameters
      * will be added to the kernel source code as preprocessor definitions. During the tuning process, tuner will generate
      * configurations for combinations of kernel parameters and their values.
      * @param id Id of kernel for which the parameter will be added.
      * @param name Name of a parameter. Parameter names for a single kernel must be unique.
      * @param values Allowed values for the parameter.
      * @param group Optional group inside which the parameter will be added. Tuning configurations are generated separately for each
      * group. This is useful when kernels contain groups of parameters that can be tuned independently. In this way, the total number
      * of generated configurations can be significantly reduced.
      */
    void AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values, const std::string& group = "");

    /** @fn void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
      * const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function)
      * Adds thread modifier function for the specified kernel. The function receives thread size in the specified dimension and
      * values of the specified kernel parameters as input and returns modified thread size based on these values. Thread modifiers
      * are useful in cases when kernel parameters affect number of required kernel threads. If multiple thread modifiers are
      * specified for the same type and dimension, they are applied in order of their addition.
      * @param id Id of kernel for which the modifier will be set.
      * @param definitionIds Kernel definitions whose thread sizes will be affected by the thread modifier.
      * @param type Type of the thread modifier. See ::ModifierType for more information.
      * @param dimension Dimension which will be affected by the modifier. See ::ModifierDimension for more information.
      * @param parameters Names of kernel parameters whose values will be passed into the modifier function. The order of parameter
      * names will correspond to the order of parameter values inside the modifier function vector argument. The corresponding
      * parameters must be added to the tuner with AddParameter() before calling this method.
      * @param function Function which receives thread size in the specified kernel dimension and values of kernel parameters as input
      * and returns modified thread size based on these values.
      */
    void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function);

    /** @fn void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
      * const ModifierDimension dimension, const std::string& parameter, const ModifierAction action)
      * Adds thread modifier function for the specified kernel. This is a simplified version of the thread modifier method which
      * supports only a single kernel parameter and limited number of actions, but is easier to use. If multiple thread modifiers
      * are specified for the same type and dimension, they are applied in order of their addition.
      * @param id Id of kernel for which the modifier will be set.
      * @param definitionIds Kernel definitions whose thread sizes will be affected by the thread modifier.
      * @param type Type of the thread modifier. See ::ModifierType for more information.
      * @param dimension Dimension which will be affected by the thread modifier. See ::ModifierDimension for more information.
      * @param parameter Name of a kernel parameter whose value will be utilized by the thread modifier. The corresponding
      * parameter must be added to the tuner with AddParameter() before calling this method.
      * @param action Action of the thread modifier. See ::ModifierAction for more information.
      */
    void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::string& parameter, const ModifierAction action);

    /** @fn void AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function)
      * Adds constraint for the specified kernel. Constraints are used to prevent generating of configurations with conflicting
      * combinations of parameter values.
      * @param id Id of kernel for which the constraint will be added.
      * @param parameters Names of kernel parameters which will be affected by the constraint function. The order of parameter
      * names corresponds to the order of parameter values inside the constraint function vector argument. Note that constraints
      * can only be added between parameters which belong into the same group. The corresponding parameters must be added to the
      * tuner with AddParameter() before calling this method.
      * @param function Function which returns true if the provided combination of parameter values is valid. Returns false otherwise.
      */
    void AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function);

    /** @fn void SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds)
      * Enables profiling of specified kernel definitions. This is useful if only some definitions inside the kernel need to be
      * profiled. By default, profiling is enabled only for the first definition specified during kernel creation. Note that this
      * method has effect only if kernel profiling functionality is enabled. See SetKernelProfiling() method for more information.
      * @param id Id of kernel for which the profiled definitions will be set.
      * @param definitionIds Ids of definitions inside the kernel for which the profiling will be enabled.
      */
    void SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds);

    /** @fn template <typename T> ArgumentId AddArgumentVector(const std::vector<T>& data, const ArgumentAccessType accessType)
      * Adds new vector argument to the tuner. Makes copy of argument data, so the source data vector remains unaffected by tuner
      * operations. Argument data will be accessed from device memory during its usage by compute API. The compute API buffer
      * will be automatically created and managed by the KTT framework.
      * @param data Kernel argument data. The data type must be trivially copyable. Bool, reference or pointer types are not supported.
      * @param accessType Access type specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentVector(const std::vector<T>& data, const ArgumentAccessType accessType);

    /** @fn template <typename T> ArgumentId AddArgumentVector(std::vector<T>& data, const ArgumentAccessType accessType,
      * const ArgumentMemoryLocation memoryLocation, const ArgumentManagementType managementType, const bool referenceUserData)
      * Adds new vector argument to the tuner. Allows wide range of argument customization options.
      * @param data Kernel argument data. The data type must be trivially copyable. Bool, reference or pointer types are not supported.
      * @param accessType Access type specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @param memoryLocation Memory location specifies whether argument data will be accessed from device or host memory during its
      * usage by compute API. See ::ArgumentMemoryLocation for more information.
      * @param managementType Management type specifies who is responsible for creating, managing data and destroying compute API buffer
      * corresponding to the argument. See ::ArgumentManagementType for more information.
      * @param referenceUserData If set to true, tuner will store reference to source data and will access it directly during buffer
      * operations. This results in lower memory overhead, but relies on a user to keep data in the source vector valid. If set to
      * false, copy of the data will be made by the tuner.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentVector(std::vector<T>& data, const ArgumentAccessType accessType, const ArgumentMemoryLocation memoryLocation,
        const ArgumentManagementType managementType, const bool referenceUserData);

    /** @fn template <typename T> ArgumentId AddArgumentVector(ComputeBuffer buffer, const size_t bufferSize,
      * const ArgumentAccessType accessType, const ArgumentMemoryLocation memoryLocation)
      * Adds new vector argument to the tuner. The argument buffer is created and managed by user and depending on the compute API,
      * can be either CUdeviceptr or cl_mem handle. The tuner will not destroy the argument.
      * @param buffer User-provided memory buffer.
      * @param bufferSize Size of the provided user buffer in bytes.
      * @param accessType Access type specifies whether argument is used for input or output. See ::ArgumentAccessType for more
      * information.
      * @param memoryLocation Memory location specifies whether argument data will be accessed from device or host memory during its
      * usage by compute API. See ::ArgumentMemoryLocation for more information.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentVector(ComputeBuffer buffer, const size_t bufferSize, const ArgumentAccessType accessType,
        const ArgumentMemoryLocation memoryLocation);

    /** @fn template <typename T> ArgumentId AddArgumentScalar(const T& data);
      * Adds new scalar argument to the tuner. All scalar arguments are read-only.
      * @param data Kernel argument data. The data type must be trivially copyable. Bool, reference or pointer types are not supported.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentScalar(const T& data);

    /** @fn template <typename T> ArgumentId AddArgumentLocal(const size_t localMemorySize)
      * Adds new local memory (shared memory in CUDA) argument to the tuner. All local memory arguments are read-only and cannot be
      * initialized from host memory. In case of CUDA API usage, local memory arguments cannot be directly set as kernel function
      * arguments. Setting a local memory argument to kernel in CUDA means that corresponding amount of memory will be allocated for
      * kernel to use. In that case, all local memory argument ids should be specified at the end of the vector when calling
      * SetArguments() method.
      * @param localMemorySize Size of kernel argument in bytes.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentLocal(const size_t localMemorySize);

    /** @fn template <typename T> ArgumentId AddArgumentSymbol(const T& data, const std::string& symbolName = "")
      * Adds new symbol argument to the tuner.
      * @param data Kernel argument data. The data type must be trivially copyable. Bool, reference or pointer types are not supported.
      * @param symbolName Name of the corresponding symbol in kernel source code. Only utilized when tuner is using CUDA API. The symbol
      * name must be unique.
      * @return Id assigned to kernel argument by tuner. The id can be used in other API methods.
      */
    template <typename T>
    ArgumentId AddArgumentSymbol(const T& data, const std::string& symbolName = "");

    /** @fn void RemoveArgument(const ArgumentId id)
      * Removes argument with the specified id from the tuner. Note that argument can only be removed if it is not associated with
      * any kernel definition.
      * @param id Id of the argument which will be removed.
      */
    void RemoveArgument(const ArgumentId id);

    /** @fn void SetReadOnlyArgumentCache(const bool flag)
      * Toggles caching of read-only kernel arguments which have management type set to framework. This can significantly speed up
      * tuning, since arguments are uploaded into compute API buffers only once. Caching is enabled by default. Users who wish to
      * modify read-only arguments inside kernel launcher may wish to disable this behaviour.
      * @param flag If true, read-only argument caching is enabled. It is disabled otherwise.
      */
    void SetReadOnlyArgumentCache(const bool flag);

    /** @fn KernelResult Run(const KernelId id, const KernelConfiguration& configuration,
      * const std::vector<BufferOutputDescriptor>& output)
      * Runs kernel using the specified configuration.
      * @param id Id of kernel which will be run.
      * @param configuration Configuration under which the kernel will be launched. See KernelConfiguration for more information.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See BufferOutputDescriptor
      * for more information.
      * @return Result containing information about kernel computation. See KernelResult for more information.
      */
    KernelResult Run(const KernelId id, const KernelConfiguration& configuration, const std::vector<BufferOutputDescriptor>& output);

    /** @fn void SetProfiling(const bool flag)
      * Toggles profiling of kernels inside the tuner. Profiled kernel runs generate profiling counters which can be used by
      * searchers and stop conditions for more accurate performance measurement. Profiling counters can also be retrieved through
      * API and saved into a file with kernel results. Note that enabling profiling will result in longer tuning times because
      * profiled kernels have to be launched multiple times with the same configuration in order to collect all profiling counters.
      * Asynchronous kernel launches are not supported when kernel profiling is enabled. Kernel profiling is disabled by default.
      * @param flag If true, kernel profiling is enabled. It is disabled otherwise.
      */
    void SetProfiling(const bool flag);

    /** @fn void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold)
      * Sets validation method and tolerance threshold for floating-point argument validation. Default validation method is side
      * by side comparison. Default tolerance threshold is 1e-4.
      * @param method Validation method which will be used for floating-point argument validation. See ::ValidationMethod for more
      * information.
      * @param toleranceThreshold Output validation threshold. If difference between tuned kernel output and reference output is
      * within the threshold, the tuned kernel output will be considered correct.
      */
    void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);

    /** @fn void SetValidationMode(const ValidationMode mode)
      * Sets mode under which kernel output validation is enabled. By default, output validation is enabled only during kernel
      * tuning.
      * @param mode Bitfield of modes under which kernel output validation is enabled. See ::ValidationMode for more information.
      */
    void SetValidationMode(const ValidationMode mode);

    /** @fn void SetValidationRange(const ArgumentId id, const size_t range)
      * Sets validation range for the specified argument. The entire argument is validated by default.
      * @param id Id of argument for which the validation range will be set. Only not read-only vector arguments can be validated.
      * @param range Number of argument elements which will be validated, starting from the first element.
      */
    void SetValidationRange(const ArgumentId id, const size_t range);

    /** @fn void SetValueComparator(const ArgumentId id, ValueComparator comparator);
      * Sets value comparator for the specified kernel argument. Arguments with custom data type cannot be compared using built-in
      * comparison operators and require user to provide a comparator. Comparator can also be optionally added for arguments with
      * built-in data types.
      * @param id Id of argument for which the comparator will be set. Only not read-only vector arguments can be validated.
      * @param comparator Function which receives two elements with data type matching the type of specified kernel argument and
      * returns true if the elements are equal. Returns false otherwise.
      */
    void SetValueComparator(const ArgumentId id, ValueComparator comparator);

    /** @fn void SetReferenceComputation(const ArgumentId id, ReferenceComputation computation)
      * Sets reference computation for the specified argument. Reference computation output will be compared to tuned kernel output
      * in order to ensure correctness of computation.
      * @param id Id of argument for which reference computation will be set. Only not read-only vector arguments can be validated.
      * @param computation Function which receives memory buffer on input where it stores its computed reference result. The size
      * of buffer matches the size of kernel argument in bytes. If a custom validation range was set, the size of buffer matches
      * the specified range.
      */
    void SetReferenceComputation(const ArgumentId id, ReferenceComputation computation);

    /** @fn void SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration)
      * Sets reference kernel for the specified argument. Reference kernel output will be compared to tuned kernel output in order
      * to ensure correctness of computation. Reference kernel uses only specified configuration.
      * @param id Id of argument for which reference kernel will be set. Only not read-only vector arguments can be validated.
      * @param referenceId Id of reference kernel.
      * @param configuration Configuration under which the reference kernel will be launched to produce reference output. This is
      * useful if the kernel has a configuration which is known to produce correct results.
      */
    void SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration);

    /** @fn std::vector<KernelResult> Tune(const KernelId id)
      * Performs the tuning process for specified kernel. Creates configuration space based on combinations of provided kernel
      * parameters and constraints. The configurations will be launched in order that depends on the specified Searcher. Tuning
      * will end when all configurations are explored.
      * @param id Id of the tuned kernel.
      * @return Vector of results containing information about kernel computation in specific configuration. See KernelResult for
      * more information.
      */
    std::vector<KernelResult> Tune(const KernelId id);

    /** @fn std::vector<KernelResult> Tune(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
      * Performs the tuning process for specified kernel. Creates configuration space based on combinations of provided kernel
      * parameters and constraints. The configurations will be launched in order that depends on the specified Searcher. Tuning
      * will end either when all configurations are explored or when the specified stop condition is fulfilled.
      * @param id Id of the tuned kernel.
      * @param stopCondition Condition which decides whether to continue the tuning process. See StopCondition for more information.
      * @return Vector of results containing information about kernel computation in specific configuration. See KernelResult for
      * more information.
      */
    std::vector<KernelResult> Tune(const KernelId id, std::unique_ptr<StopCondition> stopCondition);

    /** @fn KernelResult TuneIteration(const KernelId id, const std::vector<BufferOutputDescriptor>& output,
      * const bool recomputeReference = false)
      * Performs one step of the tuning process for specified kernel. When this method is called for the kernel for the first time,
      * it creates configuration space based on combinations of provided kernel parameters and constraints. Each time this method
      * is called, it launches a single kernel configuration. If all configurations were already launched, it runs kernel using the
      * best configuration. Output data can be retrieved by providing output descriptors. Allows control over recomputation of
      * reference output.
      * @param id Id of the tuned kernel.
      * @param output User-provided memory locations for kernel arguments which should be retrieved. See BufferOutputDescriptor for
      * more information.
      * @param recomputeReference Flag which controls whether recomputation of reference output should be performed or not. Useful
      * if kernel data between individual method invocations change.
      * @return Result containing information about kernel computation in specific configuration. See KernelResult for more
      * information.
      */
    KernelResult TuneIteration(const KernelId id, const std::vector<BufferOutputDescriptor>& output,
        const bool recomputeReference = false);

    /** @fn std::vector<KernelResult> SimulateKernelTuning(const KernelId id, const std::vector<KernelResult>& results,
      * const uint64_t iterations = 0)
      * Performs simulated tuning process for the specified kernel. The kernel is not tuned, execution times are read from the
      * provided results. Creates configuration space based on combinations of provided kernel parameters and constraints. The
      * configurations will be launched in order that depends on specified Searcher. This method can be used to test behaviour
      * and performance of newly implemented searchers. The provided results should correspond to the results output by the same
      * kernel during regular tuning.
      * @param id Id of the kernel for simulated tuning.
      * @param results Results from which the kernel execution times will be retrieved.
      * @param iterations Number of iterations performed. If equal to 0, search of the entire tuning space is performed.
      * @return Vector of results for configurations chosen by the searcher during simulated tuning.
      */
    std::vector<KernelResult> SimulateKernelTuning(const KernelId id, const std::vector<KernelResult>& results,
        const uint64_t iterations = 0);

    /** @fn void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
      * Sets searcher which will be used during kernel tuning. If no searcher is specified, DeterministicSearcher will be used.
      * @param id Id of kernel for which searcher will be set.
      * @param searcher Searcher which decides which kernel configuration will be launched next. See Searcher for more information.
      */
    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);

    /** @fn void ClearData(const KernelId id)
      * Resets tuning process and clears generated configurations for the specified kernel.
      * @param id Id of kernel whose data will be cleared.
      */
    void ClearData(const KernelId id);

    /** @fn KernelConfiguration GetBestConfiguration(const KernelId id) const
      * Returns the best configuration found for specified kernel. Valid configuration will be returned only if kernel tuning was
      * already performed for the corresponding kernel.
      * @param id Id of kernel for which the best configuration will be returned.
      * @return Best configuration for the specified kernel. See KernelConfiguration for more information.
      */
    KernelConfiguration GetBestConfiguration(const KernelId id) const;

    /** @fn KernelConfiguration CreateConfiguration(const KernelId id, const ParameterInput& parameters) const
      * Creates and returns configuration for the specified kernel based on provided parameters and their values.
      * @param id Id of kernel for which the configuration will be created.
      * @param parameters Vector of parameter names and their values from which the configuration is generated. If certain
      * parameters are omitted, their first specified values are added to the configuration.
      * @return Configuration created based on the specified input. See KernelConfiguration for more information.
      */
    KernelConfiguration CreateConfiguration(const KernelId id, const ParameterInput& parameters) const;

    /** @fn std::string GetKernelSource(const KernelId id, const KernelConfiguration& configuration) const
      * Returns kernel source with preprocessor definitions for the specified kernel based on provided configuration. Valid source
      * is returned only for kernels with single definition.
      * @param id Id of kernel for which the source is returned.
      * @param configuration Kernel configuration for which the source will be generated. See KernelConfiguration for more
      * information.
      * @return Kernel source with preprocessor definitions for the specified kernel based on provided configuration.
      */
    std::string GetKernelSource(const KernelId id, const KernelConfiguration& configuration) const;

    /** @fn std::string GetKernelDefinitionSource(const KernelDefinitionId id, const KernelConfiguration& configuration) const
      * Returns kernel source with preprocessor definitions for the specified kernel definition based on provided configuration.
      * @param id Id of kernel definition for which the source is returned.
      * @param configuration Kernel configuration for which the source will be generated. See KernelConfiguration for more
      * information.
      * @return Kernel definition source with preprocessor definitions for the specified kernel based on provided configuration.
      */
    std::string GetKernelDefinitionSource(const KernelDefinitionId id, const KernelConfiguration& configuration) const;

    /** @fn static void SetTimeUnit(const TimeUnit unit)
      * Sets time unit used for printing of results. Default time unit is milliseconds.
      * @param unit Time unit which will be used for printing of results. See ::TimeUnit for more information.
      */
    static void SetTimeUnit(const TimeUnit unit);

    /** @fn void SaveResults(const std::vector<KernelResult>& results, const std::string& filePath, const OutputFormat format,
      * const UserData& data = {}) const
      * Saves specified kernel results to the specified file.
      * @param results Results which will be saved.
      * @param filePath File where the results will be saved. The file extension is added automatically based on the specified
      * format.
      * @param format Format in which the results are saved. See ::OutputFormat for more information.
      * @param data User data which will be saved into the file together with results.
      */
    void SaveResults(const std::vector<KernelResult>& results, const std::string& filePath, const OutputFormat format,
        const UserData& data = {}) const;

    /** @fn std::vector<KernelResult> LoadResults(const std::string& filePath, const OutputFormat format) const
      * Loads kernel results from the specified file. The file must be previously created by the tuner method SaveResults() with
      * corresponding output format.
      * @param filePath File from which the results will be loaded. The file extension is added automatically based on the
      * specified format.
      * @param format Format in which the results are stored. See ::OutputFormat for more information.
      * @return Results loaded from the file.
      */
    std::vector<KernelResult> LoadResults(const std::string& filePath, const OutputFormat format) const;

    /** @fn std::vector<KernelResult> LoadResults(const std::string& filePath, const OutputFormat format, UserData& data) const
      * Loads kernel results from the specified file. The file must be previously created by the tuner method SaveResults() with
      * corresponding output format.
      * @param filePath File from which the results will be loaded. The file extension is added automatically based on the
      * specified format.
      * @param format Format in which the results are stored. See ::OutputFormat for more information.
      * @param data User data which will be loaded from the file together with results.
      * @return Results loaded from the file.
      */
    std::vector<KernelResult> LoadResults(const std::string& filePath, const OutputFormat format, UserData& data) const;

    /** @fn void Synchronize()
      * Blocks until all commands submitted to all KTT device queues are completed.
      */
    void Synchronize();

    /** @fn void SetProfilingCounters(const std::vector<std::string>& counters)
      * Specifies profiling counters that will be collected during kernel profiling. Note that not all profiling counters are
      * available on all devices.
      * For the list of old CUDA CUPTI profiling counters, see: https://docs.nvidia.com/cupti/Cupti/r_main.html#metrics-reference
      * For the list of new CUDA CUPTI profiling counters, see: https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_raw_metrics_api
      * For the list of AMD GPA profiling counters, see: https://gpuperfapi.readthedocs.io/en/latest/counters.html
      * @param counters Names of counters that will be collected during kernel profiling.
      */
    void SetProfilingCounters(const std::vector<std::string>& counters);

    /** @fn void SetCompilerOptions(const std::string& options)
      * Sets compute API compiler options to specified options. There are no default options for OpenCL backend. By default for
      * CUDA backend it adds the compiler option "--gpu-architecture=compute_xx", where `xx` is the compute capability retrieved
      * from the device.
      * For the list of OpenCL compiler options, see: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clBuildProgram.html
      * For the list of CUDA compiler options, see: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options
      * @param options Compute API compiler options. If multiple options are used, they need to be separated by a single space character.
      */
    void SetCompilerOptions(const std::string& options);

    /** @fn void SetGlobalSizeType(const GlobalSizeType type)
      * Sets global size specification type to specified compute API style. In OpenCL, NDrange size is specified as number
      * of work-items in a work-group multiplied by number of work-groups. In CUDA, grid size is specified as number of blocks.
      * This method makes it possible to use OpenCL style in CUDA and vice versa. Default global size type is the one corresponding
      * to the compute API used by the tuner.
      * @param type Global size type which will be set for tuner. See ::GlobalSizeType for more information.
      */
    void SetGlobalSizeType(const GlobalSizeType type);

    /** @fn void SetAutomaticGlobalSizeCorrection(const bool flag)
      * Toggles automatic correction for kernel global size, which ensures that global size in each dimension is always a multiple
      * of local size in corresponding dimension. Performs a roundup to the nearest higher multiple. Automatic global size correction
      * is disabled by default. Note that automatic global size correction works only if global size type is set to OpenCL.
      * @param flag If true, automatic global size correction will be enabled. It will be disabled otherwise.
      */
    void SetAutomaticGlobalSizeCorrection(const bool flag);

    /** @fn void SetKernelCacheCapacity(const uint64_t capacity)
      * Sets capacity of compiled kernel cache used by the tuner. The cache contains recently compiled kernels which are prepared
      * to be launched immediately, eliminating compilation overhead. Using the cache can significantly improve tuner performance
      * during online tuning or iterative kernel running with custom KernelLauncher. Default cache size is 10.
      * @param capacity Controls kernel cache capacity. If zero, kernel cache is completely disabled.
      */
    void SetKernelCacheCapacity(const uint64_t capacity);

    /** @fn std::vector<PlatformInfo> GetPlatformInfo() const
      * Retrieves detailed information about all available platforms. See PlatformInfo for more information.
      * @return Information about all available platforms.
      */
    std::vector<PlatformInfo> GetPlatformInfo() const;

    /** @fn std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platform) const
      * Retrieves detailed information about all available devices on the specified platform. See DeviceInfo for more information.
      * @param platform Index of platform for which the device information will be retrieved.
      * @return Information about all available devices on the specified platform.
      */
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platform) const;

    /** @fn DeviceInfo GetCurrentDeviceInfo() const
      * Retrieves detailed information about device used by the tuner. See DeviceInfo for more information.
      * @return Information about device used by the tuner.
      */
    DeviceInfo GetCurrentDeviceInfo() const;

     /** @fn static void SetLoggingLevel(const LoggingLevel level)
      * Sets logging level for tuner. Default logging level is info.
      * @param level Logging level which will be used by tuner. See ::LoggingLevel for more information.
      */
    static void SetLoggingLevel(const LoggingLevel level);

    /** @fn static void SetLoggingTarget(std::ostream& outputTarget)
      * Sets the target for info messages logging to specified output stream. Default logging target is `std::clog`.
      * @param outputTarget Location where tuner info messages will be printed.
      */
    static void SetLoggingTarget(std::ostream& outputTarget);

    /** @fn static void SetLoggingTarget(const std::string& filePath)
      * Sets the target for info messages logging to specified file. Default logging target is `std::clog`.
      * @param filePath Path to file where tuner info messages will printed.
      */
    static void SetLoggingTarget(const std::string& filePath);

private:
    std::unique_ptr<TunerCore> m_Tuner;

    ArgumentId AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, void* data, const size_t dataSize);
    ArgumentId AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const void* data, const size_t dataSize, const std::string& symbolName = "");
    ArgumentId AddUserArgument(ComputeBuffer buffer, const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize);

    template <typename T>
    ArgumentDataType DeriveArgumentDataType() const;
};

} // namespace ktt

#include <Tuner.inl>
