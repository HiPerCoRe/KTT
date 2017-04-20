KTT API Documentation
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

* `void printComputeAPIInfo(std::ostream& outputTarget)`:
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
    
Kernel tuning methods
---------------------

* `void tuneKernel(const size_t kernelId)`:
Starts the tuning process for specified kernel.

Result printing methods
-----------------------

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
