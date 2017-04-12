KTT API Documentation
=====================

This file describes the API of KTT framework. All classes and methods are located in the `ktt` namespace.

Constructor
-----------

* `Tuner(const size_t platformIndex, const size_t deviceIndex)`:
Creates new tuner object for specified platform and device. Indices for all available
platforms and devices can be retrieved by calling `printComputeAPIInfo()` method

Compute API methods
-------------------

* `void setCompilerOptions(const std::string& options)`:
Sets compute API compiler options to those provided in argument.

* `void printComputeAPIInfo(std::ostream& outputTarget)`:
Prints basic information about available plaforms and devices, including indices
assigned by KTT framework, to specified output stream.

* `std::vector<PlatformInfo> getPlatformInfo()`:
Retrieves list of objects containing detailed information about all available platforms.

* `std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex)`:
Retrieves list of objects containing detailed information about all available devices on specified platform.

Kernel handling methods
-----------------------

* `size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from string. Requires specification of kernel name (matching kernel name inside source), global and local thread size.
Returns id assigned to kernel by tuner.

* `size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from file. Requires specification of kernel name (matching kernel name inside source), global and local thread size.
Returns id assigned to kernel by tuner.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)`:
Adds new parameter for specified kernel, parameter needs to have unique name and list of values.
During tuning, parameter definitions will be added to kernel source as `#define PARAMETER_NAME PARAMETER_VALUE`.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)`:
Similar as above, but this time parameter value also modifies number of threads in either global or local space in specified dimension.
Form of modification depends on thread modifier action argument. If there are multiple thread modifiers for same space and dimension, modification is applied
in the order of parameters' addition.

* `void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction, const std::vector<std::string>& parameterNames)`:
Adds new constraint for specified kernel. Constraints are used to prevent generating of invalid configurations (eg. conflicting parameter values).

* `void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Sets kernel arguments for specified kernel by providing corresponding argument ids. Different kernels can have same
arguments assigned (copies of arguments for each kernel will be made). Argument ids must be specified in order of their declaration in kernel source.

* `void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)`:
Specifies search method for given kernel. List of search arguments needs to contain 0, 1, 2 or 5 elements, based on chosen search method.
Default search method is full search. 

Argument handling methods
-------------------------

* `size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)`:
Adds new vector argument to kernel. Argument memory type specifies whether argument is used for input or output (or both).
Currently supported data types are double, float and int. Returns id assigned to argument by tuner.

* `size_t addArgument(const T value)`:
Adds new scalar argument to kernel. Currently supported data types are double, float and int.
Returns id assigned to argument by tuner.
    
Kernel tuning methods
---------------------

* `void tuneKernel(const size_t kernelId)`:
Starts the autotuning process for specified kernel.

Result printing methods
-----------------------

* `void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given output stream. Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

* `void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const`:
Prints tuning results for specified kernel to given file. Valid results will be printed only if method `tuneKernel()` was already called for corresponding kernel.

Result validation methods
-------------------------

* `void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds)`:
Sets reference kernel for specified kernel. Reference kernel output will be compared to tuned kernel output in order to ensure correctness of computation.
Reference kernel can be the same as tuned kernel (reference kernel only uses single configuration). Only specified output arguments will be validated.

* `void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds)`:
Sets reference class for specified kernel. Reference class output will be compared to tuned kernel output in order to ensure correctness of computation.
Only specified output arguments will be validated.

* `void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)`:
Sets validation method and tolerance threshold for floating point arguments. Default validation method is side by side comparison. Default tolerance threshold is 1e-4.
