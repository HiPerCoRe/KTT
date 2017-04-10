KTT: API reference
==================

This file describes the API of the KTT framework. Everything is located in the `ktt` namespace.

Constructor
-----------

* `Tuner(const size_t platformIndex, const size_t deviceIndex)`:
Creates new tuner object for specified platform and device. Indices for all available
platforms and devices can be retrieved by calling `printComputeAPIInfo()` method

Compute API methods
-------------------

* `void setCompilerOptions(const std::string& options)`:
Sets compute API compiler options to options provided in argument.

* `void printComputeAPIInfo(std::ostream& outputTarget)`:
Prints basic information about available plaforms and devices, including indices
assigned by KTT framework.

* `std::vector<PlatformInfo> getPlatformInfo()`:
Retrieves list of objects containing detailed information about all available platforms.

* `std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex)`:
Retrieves list of objects containing detailed information about all available devices.

Kernel handling methods
-----------------------

* `size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from string. Requires specification of kernel name (which mathces kernel name inside .cl file), NDRange size and workgroup size.
Returns id assigned to kernel by tuner.

* `size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize)`:
Adds new kernel to tuner from file. Requires specification of kernel name (which mathces kernel name inside .cl file), NDRange size and workgroup size.
Returns id assigned to kernel by tuner.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values)`:
Adds new parameter to specified kernel, parameter needs to have unique name and list of values.
During tuning, parameter definitions will be added to source file as `#define PARAMETER_NAME PARAMETER_VALUE`.

* `void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)`:
Similar as above, but this time, parameter value modifies number of threads in either NDRange or workgroup.

* `void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames)`:
todo

* `void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)`:
Sets kernel arguments for specified kernel by providing argument ids. Different kernels can have same
arguments assigned (copies of arguments for each kernel will be made).

* `void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)`:
todo

Argument handling methods
-------------------------

* `size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)`:
todo

* `size_t addArgument(const T value)`:
todo

* `void updateArgument(const size_t argumentId, const std::vector<T>& data)`:
todo

* `void updateArgument(const size_t argumentId, const T value)`:
todo
    
Kernel tuning methods
---------------------

* `void tuneKernel(const size_t kernelId)`:
Starts the autotuning process for specified kernel.

Result printing methods
-----------------------

* `void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const`:
todo

* `void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const`:
todo

Result validation methods
-------------------------

* `void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)`:
todo

* `void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds)`:
todo

* `void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const size_t resultArgumentId)`:
todo
