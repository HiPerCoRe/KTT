KTT API documentation
=====================

This file describes the API of KTT framework. All classes and methods are located in the `ktt` namespace.

Constructors
------------

* `Tuner(const size_t platformIndex, const size_t deviceIndex)`:
Creates new tuner object for specified platform and device.
Tuner uses OpenCL as compute API and operates in tuning mode.
Indices for all available platforms and devices can be retrieved by calling `printComputeApiInfo()` method.

* `Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi)`:
Similar to previous constructor, but also allows choice of compute API.
If selected compute API is Nvidia CUDA, platform index is ignored.

* `Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode)`:
Similar to previous constructor, but also allows choice of run mode.
Two different run modes are supported.

In tuning mode, all kernel arguments are copied inside tuner when argument addition methods are called.
Additionally, extra argument copies are created for each kernel launch under different configuration.
It is possible to perform output validation.

In computation mode, tuner directly uses buffers provided by client application.
It is furthermore possible to perform argument zero-copy by using appropriate argument flags, which results in output
data being written directly to client buffers.
Output validation is disabled and calling any method related to validation will result in an exception.

Compute API methods
-------------------

* `void setCompilerOptions(const std::string& options)`:
Sets compute API compiler options to specified options.
Individual options have to be separated by a single space character.
Default options string for OpenCL back-end is empty.
Default options string for CUDA back-end is "--gpu-architecture=compute_30".

* `void printComputeApiInfo(std::ostream& outputTarget)`:
Prints basic information about available platforms and devices, including indices assigned by tuner, to specified output stream.

* `std::vector<PlatformInfo> getPlatformInfo()`:
Retrieves list of objects containing detailed information about all available platforms (such as platform name, vendor, list of extensions, etc.).
PlatformInfo object supports output operator.

* `std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex)`:
Retrieves list of objects containing detailed information about all available devices (such as device name, memory sizes, list of extensions, etc.) on specified platform.
DeviceInfo object supports output operator.

* `DeviceInfo getCurrentDeviceInfo()`:
Retrieves object containing detailed information about currently used device (such as device name, memory sizes, list of extensions, etc.).

Basic kernel handling methods
-----------------------------

* `size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from source inside string. Requires specification of kernel name (matching the one inside kernel source) and default global / local thread sizes.
Returns id assigned to kernel by tuner.

* `size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Similar to previous method, but loads kernel source from a file.

* `void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Sets kernel arguments for specified kernel by providing corresponding argument ids returned by argument addition methods.
Different kernels can have same arguments assigned. Copies of arguments for each kernel will be made during the tuning process.
Argument ids must be specified in order of their declaration inside kernel source.
Argument ids must be unique.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)`:
Adds new parameter for specified kernel, parameter needs to have a unique name and list of valid values.
During the tuning process, parameter definitions will be added to kernel source as `#define PARAMETER_NAME PARAMETER_VALUE`.

Advanced kernel handling methods
--------------------------------

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)`:
Similar to previous method, but also allows the parameter to act as thread size modifier.
Parameter value modifies number of threads in either global or local space in specified dimension.
Form of modification depends on thread modifier action argument. If there are multiple thread modifiers present for same space and dimension, actions are applied in the order of parameters' addition.

* `void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction, const std::vector<std::string>& parameterNames)`:
Adds new constraint for specified kernel. Constraints are used to prevent generating of invalid configurations (eg. conflicting parameter values).

* `void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)`:
Sets tuning manipulator for specified kernel.
Tuning manipulator enables customization of kernel execution by providing specialized method for computation.
Specialized method can, for example, run part of the computation directly in C++ code, utilize iterative kernel launches, etc.

Composition handling methods
----------------------------
* `size_t addKernelComposition(const std::string& compositionName, const std::vector<size_t> kernelIds, std::unique_ptr<TuningManipulator> tuningManipulator)`:
Creates a kernel composition from specified kernels.
Following regular kernel methods can also be applied on kernel composition and will call corresponding method for all kernels inside the composition: `setKernelArguments()`, `addParameter()` (both versions), `addConstraint()`.
Kernel compositions do not inherit any parameters or constraints from the original kernels.
Adding parameters or constraints to kernels inside given composition will not affect the original kernels or other compositions.
Tuning manipulator is required in order to launch kernel composition with tuner.
Composition name is used during output printing.

* `void addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)`:
Calls thread modifier version of `addParameter()` method for a single kernel inside specified kernel composition.

* `void setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Calls `setKernelArguments()` method for a single kernel inside specified kernel composition.

Argument handling methods
-------------------------

* `size_t addArgument(const std::vector<T>& data, const ArgumentAccessType& accessType)`:
Adds new vector argument to tuner.
During usage, argument will be copied to device memory.
Argument access type specifies whether argument is used for input or output (or both).
Supported data type sizes are 8, 16, 32 and 64 bits. Provided data type must be trivially copyable.
Returns id assigned to argument by tuner.

* `size_t addArgument(const std::vector<T>& data, const ArgumentAccessType& accessType, const ArgumentMemoryLocation& memoryLocation)`:
Similar to previous method, but also allows choice of argument memory location.
Argument memory location specifies whether argument will be copied to device or host memory during its usage.
If `ArgumentMemoryLocation::HostZeroCopy` is selected and tuner is in computation mode, client buffer will be used
directly for input and output. In certain scenarios, this will prevent any argument copies from being made.
In tuning mode, this flag acts in the same way as `ArgumentMemoryLocation::Host`.

* `size_t addArgument(const T& value)`:
Adds new scalar argument to tuner. All scalar arguments are read-only.
Supported data type sizes are 8, 16, 32 and 64 bits. Provided data type must be trivially copyable.
Returns id assigned to argument by tuner.

* `size_t addArgument(const size_t elementsCount)`:
Adds new local memory argument to tuner. All local memory arguments are read-only.
Elements count specifies, how many elements of provided data type will the argument contain.
Supported data type sizes are 8, 16, 32 and 64 bits. Provided data type must be trivially copyable.
Returns id assigned to argument by tuner.

* `void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)`:
Enables printing of specified output argument to specified file.
It is possible to specify whether to print only valid, invalid or all arguments.
It is not recommended to enable argument printing for very large arguments.

Kernel launch and tuning methods
--------------------------------

* `void tuneKernel(const size_t kernelId)`:
Starts the tuning process for specified kernel.

* `void runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration, const std::vector<ArgumentOutputDescriptor>& outputDescriptors)`:
Runs specified kernel using provided configuration.
Output arguments can be retrieved by providing output descriptors.
Each output descriptor contains id of an argument to be retrieved, memory location where data will be written and optionally
size of the data if only part of an argument needs to be retrieved.
Target memory location's size has to be equal or greater than size of retrieved data.
No result validation is performed.

* `void setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)`:
Specifies search method used during kernel tuning. Number of required search arguments depends on the search method.
Default search method is full search, which requires no search arguments.
Other methods require following search arguments:
    - Random search - (0) fraction
    - PSO - (0) fraction, (1) swarm size, (2) global influence, (3) local influence, (4) random influence
    - Annealing - (0) fraction, (1) maximum temperature

    Fraction argument specifies how many configurations out of all configurations will be explored during the tuning process (eg. setting fraction to 0.5 will cause tuner to explore half of the configurations).
    Swarm size argument will be converted to size_t.

Result retrieval methods
------------------------

* `void setPrintingTimeUnit(const TimeUnit& timeUnit)`:
Sets time unit used during printing of results to specified unit.
This only affects `printResult()` methods. Default time unit is microseconds. 

* `void setInvalidResultPrinting(const bool flag)`:
Toggles printing of results from failed kernel runs.
Invalid results will be separated from valid results during printing.
Printing of invalid results is disabled by default.

* `void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given output stream.
Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

* `void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given file.
Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

* `std::vector<ParameterValue> getBestConfiguration(const size_t kernelId) const`:
Returns best configuration for specified kernel.
Valid configuration will be returned only if method `tuneKernel()` was already called for corresponding kernel.

Result validation methods
-------------------------

* `void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)`:
Sets reference kernel for specified kernel.
Reference kernel output will be compared to tuned kernel output in order to ensure correctness of computation.
Reference kernel uses only single configuration, it cannot be composite and cannot use tuning manipulator. 
Only specified output arguments will be validated.

* `void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds)`:
Sets reference class for specified kernel.
Reference class output will be compared to tuned kernel output in order to ensure correctness of computation.
Only specified output arguments will be validated.

* `void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)`:
Sets validation method and tolerance threshold for floating point arguments.
Default validation method is side by side comparison. Default tolerance threshold is 1e-4.

* `void setValidationRange(const size_t argumentId, const size_t validationRange)`:
Sets validation range for specified argument to given validation range.
Only elements within validation range, starting with first element, will be validated.
By default, all elements of an argument are validated.

Utility methods
---------------

* `void setAutomaticGlobalSizeCorrection(const bool flag)`:
Toggles automatic correction for global size, which ensures that global size in each dimension is always a multiple of local size in corresponding dimension.
Performs a roundup to the nearest higher multiple.
Automatic global size correction is turned off by default.

* `void setGlobalSizeType(const GlobalSizeType& globalSizeType)`:
Sets global size specification type to specified compute API style.
In OpenCL, NDrange size is specified as number of work-items in work-group * number of work-groups.
In CUDA, grid size is specified as number of threads in block / number of blocks.
This method makes it possible to use OpenCL style in CUDA and vice versa.

* `void setLoggingTarget(std::ostream& outputTarget)`:
Sets target for info messages logging to specified output stream.

* `void setLoggingTarget(const std::string& filePath)`:
Sets target for info messages logging to specified file.

Reference class usage
=====================

In order to use reference class for result validation, new class, which publicly inherits from ReferenceClass must be created.
ReferenceClass contains following public methods:

* `~ReferenceClass()`:
Inheriting class can override destructor with custom implementation if needed.
Default implementation is provided by API.

* `void computeResult()`:
Inheriting class must provide implementation for this method.
Reference results for all validated arguments must be computed inside this method and stored for later retrieval by tuner.

* `void* getData(const size_t argumentId) const`:
Inheriting class must provide implementation for this method.
Returns pointer to buffer containing reference result for specified validated argument.
This method will only be called after running `computeResult()`.

* `size_t getNumberOfElements(const size_t argumentId) const`:
Inheriting class can override this method, which is useful in conjuction with `setValidationRange()` method.
Returns number of elements returned by `getData()` method for specified validated argument.
This method will only be called after running `computeResult()`.

Tuning manipulator usage
========================

In order to use tuning manipulator, new class, which publicly inherits from TuningManipulator class must be created.
TuningManipulator class contains following public methods:

* `~TuningManipulator()`:
Inheriting class can override destructor with custom implementation if needed.
Default implementation is provided by API.

* `void launchComputation(const size_t kernelId)`:
Inheriting class must provide implementation for this method. Provided argument is an id of currently tuned kernel.
This method must, at very least, call `runKernel()` method with provided kernel id as its first argument.
This method can also call any other methods available in base TuningManipulator class.

* `void runKernel(const size_t kernelId)`:
Launches kernel with specified id, using thread sizes based only on the current configuration.
Provided kernel id must be either id of main kernel or one of ids returned by `getUtilizedKernelIds()` method.

* `void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Launches kernel with specified id, using specified thread sizes.
Provided kernel id must be either id of main kernel or one of ids returned by `getUtilizedKernelIds()` method.

* `DimensionVector getCurrentGlobalSize(const size_t kernelId) const`:
Returns global thread size of specified kernel based on the current configuration.
Provided kernel id must be either id of main kernel or one of ids returned by `getUtilizedKernelIds()` method.

* `DimensionVector getCurrentLocalSize(const size_t kernelId) const`:
Returns local thread size of specified kernel based on the current configuration.
Provided kernel id must be either id of main kernel or one of ids returned by `getUtilizedKernelIds()` method.

* `std::vector<ParameterValue> getCurrentConfiguration() const`:
Returns configuration used inside current run of `launchComputation()` method.

* `void updateArgumentScalar(const size_t argumentId, const void* argumentData)`:
Updates specified scalar argument.
This method only affects run of `launchComputation()` method under current configuration.
This method is useful for iterative kernel launches.

* `void updateArgumentLocal(const size_t argumentId, const size_t numberOfElements)`:
Updates specified local memory argument.
This method only affects run of `launchComputation()` method under current configuration.
This method is useful for iterative kernel launches.

* `void updateArgumentVector(const size_t argumentId, const void* argumentData)`:
Updates specified vector argument.
This method only affects run of `launchComputation()` method under current configuration.
This method is useful for iterative kernel launches.

* `void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements)`:
Updates specified vector argument.
Possibly also modifies number of elements inside the argument.
This method only affects run of `launchComputation()` method under current configuration.
This method is useful for iterative kernel launches.

* `void getArgumentVector(const size_t argumentId, void* destination) const`:
Retrieves specified vector argument.
Destination buffer size needs to be equal or greater than argument size.
This method is useful for iterative kernel launches.

* `void getArgumentVector(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const`:
Retrieves part of specified vector argument.
Destination buffer size needs to be equal or greater than specified data size.
This method is useful for iterative kernel launches.

* `void changeKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Sets kernel arguments for specified kernel by providing corresponding argument ids.
Argument ids must be unique.
This method only affects run of `launchComputation()` method under current configuration.

* `void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond)`:
Swaps positions of specified kernel arguments for specified kernel.
This method only affects run of `launchComputation()` method under current configuration.

* `std::vector<size_t> convertFromDimensionVector(const DimensionVector& vector)`:
Converts provided dimension vector to standard vector.

* `DimensionVector convertToDimensionVector(const std::vector<size_t>& vector)`:
Converts provided standard vector to dimension vector.
If provided vector size is less than 3, fills remaining dimension vector positions with 1s.

* `size_t getParameterValue(const std::string& parameterName, const std::vector<ParameterValue>& parameterValues)`:
Returns value of specified parameter from provided list of parameters.

Default tuning manipulator implementation
-----------------------------------------

Following example shows how default tuning manipulator implementation looks like (no difference in functionality compared to tuning of kernel without using a manipulator):
```c++
class SimpleTuningManipulator : public ktt::TuningManipulator
{
public:
    void launchComputation(const size_t kernelId) override
    {
        runKernel(kernelId);
    }
};
```
