KTT API documentation
=====================

This file describes the API of KTT framework. All classes and methods are located in the `ktt` namespace.

Constructor
-----------

* `Tuner(const size_t platformIndex, const size_t deviceIndex)`:
Creates new tuner object for specified platform and device.
Indices for all available platforms and devices can be retrieved by calling `printComputeAPIInfo()` method.

Compute API methods
-------------------

* `void setCompilerOptions(const std::string& options)`:
Sets compute API compiler options to specified options.

* `void printComputeApiInfo(std::ostream& outputTarget)`:
Prints basic information about available platforms and devices, including indices assigned by KTT framework, to specified output stream.

* `std::vector<PlatformInfo> getPlatformInfo()`:
Retrieves list of objects containing detailed information about all available platforms (such as platform name, vendor, list of extensions, etc.).
PlatformInfo object supports output operator.

* `std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex)`:
Retrieves list of objects containing detailed information about all available devices (such as device name, memory sizes, list of extensions, etc.) on specified platform.
DeviceInfo object supports output operator.

Basic kernel handling methods
-----------------------------

* `size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from source inside string. Requires specification of kernel name (matching the one inside kernel source) and default global / local thread sizes.
Returns id assigned to kernel by tuner.

* `size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from source inside file. Requires specification of kernel name (matching the one inside kernel source) and default global / local thread sizes.
Returns id assigned to kernel by tuner.

* `void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Sets kernel arguments for specified kernel by providing corresponding argument ids (returned by argument addition methods).
Different kernels can have same arguments assigned (copies of arguments for each kernel will be made during the tuning process).
Argument ids must be specified in order of their declaration inside kernel source.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)`:
Adds new parameter for specified kernel, parameter needs to have a unique name and list of valid values.
During the tuning process, parameter definitions will be added to kernel source as `#define PARAMETER_NAME PARAMETER_VALUE`.

Advanced kernel handling methods
--------------------------------

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)`:
Adds new parameter for specified kernel, parameter needs to have a unique name and list of valid values.
During the tuning process, parameter definitions will be added to kernel source as `#define PARAMETER_NAME PARAMETER_VALUE`.
Additionally, parameter value modifies number of threads in either global or local space in specified dimension.
Form of modification depends on thread modifier action argument. If there are multiple thread modifiers present for same space and dimension, actions are applied in the order of parameters' addition.

* `void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction, const std::vector<std::string>& parameterNames)`:
Adds new constraint for specified kernel. Constraints are used to prevent generating of invalid configurations (eg. conflicting parameter values).

* `void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)`:
Specifies search method for given kernel. Number of required search arguments depends on specified search method.
Default search method is full search, which requires no search arguments.
Other methods require following search arguments:
    - Random search - (0) fraction
    - PSO - (0) fraction, (1) swarm size, (2) global influence, (3) local influence, (4) random influence
    - Annealing - (0) fraction, (1) maximum temperature

    Fraction argument specifies how many configurations out of all configurations will be explored during the tuning process (eg. setting fraction to 0.5 will cause tuner to explore half of the configurations).
    Swarm size argument will be converted to size_t.

* `void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)`:
Sets tuning manipulator for specified kernel.
Tuning manipulator enables customization of kernel execution by providing specialized method for computation.
Specialized method can, for example, run part of the computation directly in C++ code, utilize iterative kernel launches, etc.

Argument handling methods
-------------------------

* `size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)`:
Adds new vector argument to kernel. Argument memory type specifies whether argument is used for input or output (or both).
Currently supported data types are double, float, int and short. Returns id assigned to argument by tuner.

* `size_t addArgument(const T value)`:
Adds new scalar argument to kernel. All scalar arguments are read-only.
Currently supported data types are double, float, int and short. Returns id assigned to argument by tuner.

* `void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)`:
Enables printing of specified output argument to specified file.
It is possible to specify to only print result arguments for kernel configurations that did not successfully pass the validation.
It is not recommended to enable argument printing for very large arguments.

Kernel tuning methods
---------------------

* `void tuneKernel(const size_t kernelId)`:
Starts the tuning process for specified kernel.

Result printing methods
-----------------------

* `void setPrintingTimeUnit(const TimeUnit& timeUnit)`:
Sets time unit used during printing of results to specified unit.
This only affects `printResult` methods. Default time unit is microseconds. 

* `void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given output stream.
Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

* `void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given file.
Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

Result validation methods
-------------------------

* `void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)`:
Sets reference kernel for specified kernel.
Reference kernel output will be compared to tuned kernel output in order to ensure correctness of computation.
Reference kernel can be the same as tuned kernel (reference kernel only uses single configuration). Only specified output arguments will be validated.

* `void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds)`:
Sets reference class for specified kernel.
Reference class output will be compared to tuned kernel output in order to ensure correctness of computation.
Only specified output arguments will be validated.

* `void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)`:
Sets validation method and tolerance threshold for floating point arguments.
Default validation method is side by side comparison. Default tolerance threshold is 1e-4.

Utility methods
---------------

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

* `ArgumentDataType getDataType(const size_t argumentId) const`:
Inheriting class must provide implementation for this method.
Returns data type of specified validated argument.
This method will only be called after running `computeResult()`.

* `size_t getDataSizeInBytes(const size_t argumentId) const`:
Inheriting class must provide implementation for this method.
Returns size of buffer (in bytes) returned by `getData()` method for corresponding validated argument.
This method will only be called after running `computeResult()`.

Tuning manipulator usage
========================

In order to use tuning manipulator, new class, which publicly inherits from TuningManipulator class must be created.
TuningManipulator class contains following public methods:

* `~TuningManipulator()`:
Inheriting class can override destructor with custom implementation if needed.
Default implementation is provided by API.

* `void launchComputation(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<ParameterValue>& parameterValues)`:
Inheriting class must provide implementation for this method. Provided arguments include id, thread sizes and parameter values for current configuration of currently tuned kernel.
Usage of these arguments is completely optional. This method must, at very least, call `runKernel()` method with currently tuned kernel id as its first argument.
This method can also call any other methods available in base TuningManipulator class.

* `std::vector<std::pair<size_t, ThreadSizeUsage>> getUtilizedKernelIds() const`:
Inheriting class must override this method in case it utilizes multiple kernels inside the `launchComputation()` method.
This method needs to return ids of all additional kernels. Id of the main kernel (specified by calling `setTuningManipulator()` method) does not need to be returned.
All additional kernels will be launched under the same configuration as main kernel, which means that they need to accept exactly the same parameters.
It is possible to specify, whether the additional kernels' thread sizes will be affected by the parameters. Main kernel's thread sizes will always be affected.

* `std::vector<ResultArgument> runKernel(const size_t kernelId)`:
Launches kernel with specified id, using thread sizes based only on the current configuration.
Returns vector of result arguments (arguments assigned to kernel with kernelId, which were tagged as input-output or output-only arguments).

* `std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Launches kernel with specified id, using specified thread sizes.
Returns vector of result arguments (arguments assigned to kernel with kernelId, which were tagged as input-output or output-only arguments).

* `void updateArgumentScalar(const size_t argumentId, const void* argumentData)`:
Updates scalar argument, which is utilized by currently tuned kernel.
This method is useful for iterative kernel launches.

* `void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes)`:
Updates vector argument, which is utilized by currently tuned kernel.
This method is useful for iterative kernel launches.

Tuning manipulator example
--------------------------

Following example shows how default tuning manipulator implementation looks like (no difference in functionality compared to tuning of kernel without using a manipulator):
```c++
class SimpleTuningManipulator : public ktt::TuningManipulator
{
public:
    virtual void launchComputation(const size_t kernelId, const ktt::DimensionVector& globalSize, const ktt::DimensionVector& localSize,
        const std::vector<ktt::ParameterValue>& parameterValues) override
    {
        runKernel(kernelId);
    }
};
```
