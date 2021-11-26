# Introduction to KTT

When optimizing performance of compute kernels, a programmer has to make a lot of decisions, such as which algorithm to
use, how to arrange data structures in memory, how to block data access to optimize caching or which factor to use for
loop unrolling. Such decisions cannot be typically made in isolation - for example, when data layout in memory is changed,
a different algorithm may perform better. Therefore, it is necessary to explore vast amount of combinations of optimization
decisions in order to reach the best performance. Moreover, the best combination of optimization decisions can differ for
various hardware devices or program setup. Therefore, a way of automatic search for the best combination of these decisions,
called autotuning, is valuable.

Naturally, in the simple use case, a batch script can be sufficient for autotuning. However, in advanced applications,
usage of an autotuning framework can be beneficial, as it can automatically handle memory objects, detect errors in autotuned
kernels or perform autotuning during program runtime.

Kernel Tuning Toolkit is a framework which allows autotuning of kernels written in CUDA, OpenCL or Vulkan. It provides unified
interface for those APIs, handles communication between host (CPU) and accelerator (GPU, Xeon Phi, etc.), checks results and
timing of tuned kernels, allows dynamic tuning during program runtime, profiling of autotuned kernels and more.

----

### Table of contents
* [Basic principles behind KTT](#basic-principles-behind-ktt)
* [Offline tuning of a single kernel](#offline-tuning-of-a-single-kernel)
* [Initialization of KTT](#initialization-of-ktt)
* [Kernel definitions and kernels](#kernel-definitions-and-kernels)
* [Kernel arguments](#kernel-arguments)
    * [Scalar arguments](#scalar-arguments)
    * [Vector arguments](#vector-arguments)
    * [Local memory arguments](#local-memory-arguments)
    * [Symbol arguments](#symbol-arguments)
* [Tuning parameters](#tuning-parameters)
    * [Parameter constraints](#parameter-constraints)
    * [Parameter groups](#parameter-groups)
    * [Thread modifiers](#thread-modifiers)
* [Output validation](#output-validation)
    * [Reference computation](#reference-computation)
    * [Reference kernel](#reference-kernel)
    * [Validation customization](#validation-customization)
* [Kernel launchers](#kernel-launchers)
* [Kernel running and tuning modes](#kernel-running-and-tuning-modes)
    * [Offline tuning](#offline-tuning)
    * [Online tuning](#online-tuning)
----

### Basic principles behind KTT

When leveraging autotuning, a programmer needs to think about which properties of their computation can be autotuned. For
example, an algorithm may contain for loop which can be unrolled. There are multiple options for unroll factor value
of this loop, e.g., 1 (no unroll), 2, 4, 8. Picking the optimal value for a given device manually is difficult, therefore
we can define a tuning parameter for the unroll factor with the specified values. Afterwards, we can launch four different
versions of our computation to see which value performs best.

In practice, the computations are often complex enough to contain multiple parts which can be optimized in this way, leading
to definition of multiple tuning parameters. For example we may have the previously mentioned loop unroll parameter with
values {1, 2, 4, 8} and another parameter controlling data arrangement in memory with values {0, 1}. Combinations of these
parameters now define 8 different versions of computation. One such combination is called tuning configuration. Together, all
tuning configurations define configuration space. The size of the space grows exponentially with addition of more tuning
parameters. KTT framework offers functionality to deal with this problem which will be discussed in the follow-up sections.

----

### Offline tuning of a single kernel

Offline kernel tuning is the simplest use case of KTT framework. It involves creating a kernel, specifying its arguments (data),
defining tuning parameters and then launching autotuning. During autotuning, tuning parameter values are propagated to kernel source
code in a form of preprocessor definitions. E.g., when configuration which contains parameter with name unroll_factor and value 2
is launched, the following code is added at the beginning of kernel source code: `#define unroll_factor 2`. The definitions can be used
to alter kernel functionality based on tuning parameter values.

In the code snippet below, we create a kernel definition by specifying the name of kernel function and path to its source file. We also define
its default global and local dimensions (e.g., size of ND-range and work-group in OpenCL, size of grid and block in CUDA). We use provided
kernel definition handle to create kernel. We can also specify custom name for the kernel which is used e.g., for logging purposes. Afterwards,
we can use the kernel handle to define tuning parameter and launch autotuning. The step of creating kernel definition and kernel separately may
seem redundant at first, but it plays important role during more complex use cases that will be covered later.

```cpp
const size_t numberOfElements = 1024 * 1024;
const ktt::DimensionVector globalDimensions(numberOfElements);
const ktt::DimensionVector localDimensions(64);

const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("computeStuff", "kernelFile.cl", globalDimensions, localDimensions);
const ktt::KernelId kernel = tuner.CreateSimpleKernel("TestKernel", definition);

/* Initialize kernel input and output */

tuner.AddParameter(kernel, "unroll_factor", std::vector<uint64_t>{1, 2, 4, 8});
tuner.Tune(kernel);
```

The next snippet demonstrates how our previously defined tuning parameter could be used to alter computation inside kernel.

```cpp
__kernel void computeStuff(__global float* input, int itemsPerThread, __global float* output)
{
    ...
    
    #if unroll_factor > 1
    #pragma unroll unroll_factor
    #endif
    for (int i = 0; i < itemsPerThread; i++)
    {
        // do some computation
    }
    
    ...
}
```

----

### Initialization of KTT

The first step before we can utilize KTT is creation of tuner instance. Tuner is one of the major KTT classes and implements large portion of
autotuning logic. Practically all of the KTT structures such as kernels, kernel arguments and tuning parameters are tied to a specific tuner instance.
The simplest tuner constructor requires 3 parameters - index for platform, index for device and type of compute API that will be utilized (e.g., CUDA,
OpenCL). The indices for platforms and devices are assigned by KTT - they can be retrieved through `PlatformInfo` and `DeviceInfo` structures. These
structures also contain some other useful information such as list of supported extensions, global memory size, number of available compute units and
more. Note that the assigned indices remain the same when autotuning applications are launched multiple times on the same computer. They change only
when the hardware configuration is changed (e.g., new device is added, old device is removed, device driver is reinstalled). Also note, that the indices
may not be the same across multiple compute APIs (e.g., index for the same device may be different under OpenCL and CUDA).

The code below demonstrates how information about all available OpenCL platforms and devices is retrieved from KTT. In this case, the tuner is created
for the first device on the first platform (both platform and device index is 0).

```cpp
ktt::Tuner tuner(0, 0, ktt::ComputeApi::OpenCL);

std::vector<ktt::PlatformInfo> platforms = tuner.GetPlatformInfo();

for (const auto& platform : platforms)
{
    std::cout << platform.GetString() << std::endl;
    std::vector<ktt::DeviceInfo> devices = tuner.GetDeviceInfo(platform.GetIndex());

    for (const auto& device : devices)
    {
        std::cout << device.GetString() << std::endl;
    }
}
```

----

### Kernel definitions and kernels

Before kernel can be launched via KTT, its source must be loaded into tuner. This is achieved by creating kernel definition. During its creation,
we specify kernel function name and kernel source. The source can be added either from string or from file. Next, we specify default global
(NDrange / grid) and local (work-group / block) sizes. The sizes are specified with KTT structure `DimensionVector` which supports up to three
dimensions. When a kernel is launched during tuning, the thread sizes chosen during definition creation will be used. There are ways to launch kernels
with different than default sizes which will be covered later. For CUDA API, addition of templated kernels is supported as well. When creating
definition, it is possible to specify types that should be used to instantiate kernel function from template. When we need to instantiate the same
kernel template with different types, we do that by adding multiple kernel definitions with corresponding types which are then handled independently.

Once we have kernel definitions, we can create kernels from them. It is possible to create a simple kernel which only uses one definition as well as
composite kernel which uses multiple definitions. Usage of composite kernels is useful for computations which require launching of multiple kernel
functions in order to compute the result. In this case it is also necessary to define kernel launcher which is a function that tells tuner in which
order and how many times each kernel function is launched. Kernel launchers are covered in detail in their own section.

Note that KTT terminology regarding kernel definitions and kernels differs slightly from regular compute APIs. KTT kernel definition roughly
corresponds to a single kernel function (also called kernel in e.g., OpenCL or CUDA). KTT kernel corresponds to a specific computation which uses
one or more kernel functions and for which it is possible to define tuning parameters. KTT framework allows kernel definitions to be shared across
multiple kernels (i.e., the same kernel function can be used in multiple computations).

```cpp
// Create convolution kernel, utilizes single kernel function
const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("conv", kernelFile, gridSize, blockSize);
const ktt::KernelId kernel = tuner.CreateSimpleKernel("Convolution", definition);

// Create kernel which performs radix sort, utilizes 3 separate kernel functions
const ktt::KernelDefinitionId definition0 = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, gridSize, blockSize);
const ktt::KernelDefinitionId definition1 = tuner.AddKernelDefinitionFromFile("top_scan", kernelFile, gridSize, blockSize);
const ktt::KernelDefinitionId definition2 = tuner.AddKernelDefinitionFromFile("bottom_scan", kernelFile, gridSize, blockSize);
const ktt::KernelId kernel = tuner.CreateCompositeKernel("Sort", {definition0, definition1, definition2});
```

----

### Kernel arguments

Kernel arguments define input and output of a kernel. KTT supports multiple forms of kernel arguments such as buffers, scalars and constant memory
arguments. Before argument can be assigned to kernel, its description must be given to the tuner. In case of a buffer argument, this includes the
initial data placed inside buffer before kernel is launched, its access type (read or write) and memory location from which kernel accesses the buffer
(host or device). Once the information is provided, tuner returns a handle to the argument. Through this handle, arguments can be assigned to kernel
definitions as shown in the code below. KTT supports a wide range of data types for kernel arguments, including all built-in integer and floating-point
types as well as custom types. Note however, that custom types must be trivially copyable, so it is possible to transfer the arguments into device memory.

```cpp
const size_t numberOfElements = 1024 * 1024;
std::vector<float> a(numberOfElements);
std::vector<float> b(numberOfElements);
std::vector<float> result(numberOfElements, 0.0f);

// Fill buffers with initial data before adding the arguments to the tuner.
for (size_t i = 0; i < numberOfElements; ++i)
{
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i + 1);
}

const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(3.0f);
tuner.SetArguments(definition, {aId, bId, resultId, scalarId});
```

#### Scalar arguments

Scalar arguments are straightforward to add. We simply need to specify the scalar argument value. The scalar value is copied inside the tuner,
so both lvalues and rvalues are supported.

```cpp
const float lvalueScalar = 322.0f;
const ktt::ArgumentId lvalueId = tuner.AddArgumentScalar(lvalueScalar);
const ktt::ArgumentId rvalueId = tuner.AddArgumentScalar(34);
```

#### Vector arguments

Vector arguments have more customization options available than scalars. Other than the initial data, it is possible to specify whether argument
is used for reading or writing. For read-only arguments, additional optimization is possible during offline tuning - since its contents do not
change, the buffer needs to be copied into memory only once before the first kernel configuration is launched and then remain the same for subsequent
configurations. Setting correct access type can therefore lead to better tuning performance.

Next, it is possible to decide memory location from which the buffer is accessed by kernel - the two main options are host memory and device memory.
Users may wish to choose different memory depending on the type of device used for autotuning (e.g., host memory for CPUs, device memory for
dedicated GPUs). For host memory, it is possible to use zero-copy option, which makes kernel access the argument data directly, instead of creating
separate buffer and thus reduce memory usage. For CUDA and OpenCL 2.0, there exists one additional memory location option - unified. Unified memory
buffers can be accessed both from host and kernel side, relying on device driver to take care of migrating the data automatically.

Management type option specifies whether buffer management is handled automatically by the tuner (e.g., write arguments are automatically reset
to initial state before new kernel configuration is launched, buffers are created and deleted automatically) or by the user. In some advanced cases,
users may wish to manage the buffers manually. Note however, that this requires usage of kernel launchers which will be discussed later.

The final option for vector arguments is whether the initial data provided by user should be copied inside the tuner or referenced directly. By default,
the data is copied which is safer (i.e., temporary arguments work correctly) but less memory efficient. In case the initial data is provided in form of
lvalue argument, direct reference can be used to avoid copying. This requires user to keep the initial data buffer valid during the time argument is
used by the tuner.

```cpp
std::vector<float> input1;
std::vector<float> input2;
std::vector<float> result;

/* Initialize data */

const ktt::ArgumentId copyInputId = tuner.AddArgumentVector(input1, ktt::ArgumentAccessType::ReadOnly);
const ktt::ArgumentId referenceInputId = tuner.AddArgumentVector(input2, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, true);
const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);

// Ok - copying temporary buffer.
{
    std::vector<float> temp{0.0f, 1.0f, 2.0f};
    const ktt::ArgumentId okId = tuner.AddArgumentVector(temp, ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, false);
}

// Bad - referencing temporary buffer!
{
    std::vector<float> temp{0.0f, 1.0f, 2.0f};
    const ktt::ArgumentId badId = tuner.AddArgumentVector(temp, ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, true);
}
```

#### Local memory arguments

Local (shared in CUDA terminology) memory arguments are used to allocate corresponding amount of cache-like memory which is shared accross all work-items
(threads) inside a work-group (thread block). User has to specify the data type and total size of allocated memory in bytes.

```cpp
// Allocate local memory for 4 floats and 2 integers.
const ktt::ArgumentId local1Id = tuner.AddArgumentLocal<float>(16);
const ktt::ArgumentId local2Id = tuner.AddArgumentLocal<int32_t>(8);
```

#### Symbol arguments

Symbol arguments were introduced in order to support CUDA arguments marked as `__constant__` or `__device__`. In other APIs, symbol arguments behave in
the same way as scalars since they do not require special handling. In case of CUDA, the name of symbol argument appearing inside CUDA kernel source
code has to be specified during argument addition to tuner.

```cpp
const ktt::ArgumentId symbolId = tuner.AddArgumentSymbol(42, "magicNumber");
```

----

### Tuning parameters

Tuning parameters in KTT can be either unsigned integers or floats. When defining new parameter, we need to specify its name (i.e., the name through
which it can be referenced in kernel source) and values. With addition of more tuning parameters, the size of tuning space grows exponentially as we
need to explore all parameter combinations. KTT provides two features for users to slow down the tuning space growth.

```cpp
// We add 4 different parameters, the size of tuning space is 40 (5 * 2 * 4 * 1)
tuner.AddParameter(kernel, "unroll_factor", std::vector<uint64_t>{1, 2, 4, 8, 16});
tuner.AddParameter(kernel, "use_constant_memory", std::vector<uint64_t>{0, 1});
tuner.AddParameter(kernel, "vector_type", std::vector<uint64_t>{1, 2, 4, 8});
tuner.AddParameter(kernel, "float_value", std::vector<double>{1.0});
```

#### Parameter constraints

The first option are tuning constraints. Through constraints, it is possible to tell tuner to skip generating configurations for certain combinations
of parameters. The constraint is a function which receives values for the specified parameters on input and decides whether that combination should
be launched. User can choose which parameters are evaluated by specific constraint. Note that currently, it is possible to add constraints only
between integer parameters.

```cpp
// We add 3 different parameters, the size of tuning space is 40 (5 * 2 * 4)
tuner.AddParameter(kernel, "unroll_factor", std::vector<uint64_t>{1, 2, 4, 8, 16});
tuner.AddParameter(kernel, "vectorized_soa", std::vector<uint64_t>{0, 1});
tuner.AddParameter(kernel, "vector_type", std::vector<uint64_t>{1, 2, 4, 8});

// We add constraint between 2 parameters, reducing size of tuning space from 40 to 35 (vectorized SoA is used only for vector types,
// constraint disables all configurations where vector_type == 1 and vectorized_soa == 1)
auto vectorizedSoA = [](const std::vector<uint64_t>& values) {return values[0] > 1 || values[1] != 1;}; 
tuner.AddConstraint(kernel, {"vector_type", "vectorized_soa"}, vectorizedSoA);
```

#### Parameter groups

The second option are tuning parameter groups. This option is mainly useful for composite kernels with certain tuning parameters only affecting one
kernel definition inside the kernel. For example, if we have composite kernel with 2 kernel definitions and each definition is affected by 3 parameters
(we have 6 parameters in total), and we know that each parameter only affects one specific definition, we can evaluate the two parameter groups
independently. This can greatly reduce the total number of evaluated configurations (e.g., if each of the parameters has 2 different values, total
number of configurations is 64 -- 2^6; with usage of parameter groups, it is only 16 -- 2^3 + 2^3). It is also possible to combine usage of
constraints and groups, however constraints can only be added between parameters which belong into the same group.

```cpp
// We add 4 different parameters split into 2 independent groups, reducing size of tuning space from 16 to 8
tuner.AddParameter(kernel, "a1", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "a2", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "b1", std::vector<uint64_t>{0, 1}, "group_b");
tuner.AddParameter(kernel, "b2", std::vector<uint64_t>{0, 1}, "group_b");
```

#### Thread modifiers

Some tuning parameters can affect global or local number of threads a kernel is launched with. For example we may have a parameter which affects
amount of work performed by each thread. The more work each thread does, the less (global) threads we need in total to perform computation. In KTT,
we can define such dependency via thread modifiers. The thread modifier is a function which takes a default thread size and changes it based on
values of specified tuning parameters.

When adding a new modifier, we specify kernel and its definitions whose thread sizes are affected by the modifier. Then we choose whether modifier
affects global or local size, its dimension and names of tuning parameters tied to modifier. The modifier function can be specified through enum
which supports certain simple functions such as multiplication or addition, but allows only one tuning parameter to be tied to modifier. Another
option is using a custom user function which can be more complex and support multiple tuning parameters. It is possible to create multiple thread
modifiers for the same thread type (global / local) and dimension. In that case, the modifiers will be applied in the order of their addition to
tuner. Similar to constraints, it is possible to tie only integer parameters to thread modifiers.

```cpp
tuner.AddParameter(kernel, "block_size", std::vector<uint64_t>{32, 64, 128, 256});

// block_size parameter decides the number of local threads.
tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "block_size", ktt::ModifierAction::Multiply);

// Larger block size means that the grid size should be smaller, so the total number of threads remains the same. Therefore we divide the grid
// size by block_size parameter.
tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"block_size"},
    [](const uint64_t defaultSize, const std::vector<uint64_t>& parameters)
{
    return defaultSize / parameters[0];
});
```

----

### Output validation

When developing kernels with large number of tuning parameters, it is often necessary to check whether each configuration computes the correct output.
KTT provides a way to automatically compare output from tuned kernel configurations to reference output. That means each time a kernel configuration is
finished, the contents of its output buffer are transferred into host memory and then compared to precomputed reference output. The reference can be
computed in two ways.

#### Reference computation

Reference computation is a function which computes the reference output in host code and stores the result in the buffer provided by KTT. The size of
that buffer matches the size of validated kernel output buffer. When defining reference computation, we only need to provide the function and the id
of validated output argument.

```cpp
tuner.SetReferenceComputation(resultArgument, [&a, &b](void* buffer)
{
    float* result = static_cast<float*>(buffer);

    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] + b[i];
    }
});
```

#### Reference kernel

Another option is to compute reference result with a kernel. In this case, we need to provide the id of reference kernel and the id of validated output
argument. It is possible for reference kernel to have tuning parameters as well, so there is an option to choose specific reference configuration. If
reference kernel has no parameters, empty configuration can be provided. The reference kernel id may be the same as tuned kernel.

```cpp
tuner.SetReferenceKernel(outputId, referenceKernel, ktt::KernelConfiguration());
```

#### Validation customization

There are certain ways to further customize how validation is performed. By default, the entire output buffer is validated. If validating only
a portion of the buffer is sufficient, setting a custom validation range is possible. In this case, the size of reference buffer provided by KTT
for reference computation validation will be automatically adjusted as well.

Validation works out-of-the-box for integer and floating-point argument data types. In case of floating-point arguments, it is possible to choose
validation method (e.g., comparing each element separately or summing up all elements and comparing the result) and tolerance threshold since
different kernel configurations may have different accuracy of computing floating-point output.

If arguments with user-defined types are used, it is necessary to define value comparator. Comparator is a function which receives two elements
with the specified type on input and decides whether they are equal. Value comparator can optionally be also used for integer and floating-point
data types to override the default comparison functionality.

----

### Kernel launchers

Kernel launchers enable users to customize how kernels are run inside KTT. Launcher is a function which defines what happens when kernel under
certain configuration is launched via tuner. For simple kernels, default launcher is provided by KTT. This launcher simply runs the kernel function
tied to kernel definition and waits until its finished. If a computation requires launching a kernel function multiple times, running some part in
host code or using multiple kernel functions, then user needs to define their own launcher. In case of composite kernels, defining custom launcher is
mandatory, since KTT does not know the order in which the individual kernel functions should be run.

Kernel launcher has access to low-level KTT compute interface on input. Through this interface, it is possible to launch kernel functions, modify
buffers and retrieve the current kernel configuration. This makes it possible for tuning parameters to affect computation behaviour in host code in
addition to modifying kernel behavior. The modifications done to kernel arguments and buffers inside a launcher are isolated to the specific kernel
configuration launch. Therefore, it is not necessary to reset arguments to their original values for each kernel launch, it is done automatically by
the tuner. The only exception to this is usage of user-managed vector arguments, those have to be reset manually.

```cpp
// This launcher is equivalent in functionality to the default simple kernel launcher provided by KTT.
tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
{
    interface.RunKernel(definition);
});
```

----

### Kernel running and tuning modes

KTT supports kernel tuning as well as ordinary kernel running. Running kernels via tuner is often more convenient compared to directly using certain
compute API, since a lot of boilerplate code such as compute queue management and kernel source compilation is abstracted. It is possible to specify
configuration under which the kernel is run, so the workflow where kernel is first tuned and then launched repeatedly with the best configuration is
supported. It is possible to transfer kernel output into host memory by utilizing `BufferOutputDescriptor` structure. When creating this structure,
we need to specify id of buffer that should be transferred and pointer to memory where the buffer contents should be saved. It is possible to pass
multiple such structures into kernel running method -- each structure corresponds to a single buffer that should be transferred. After kernel run is
finished, `KernelResult` structure is returned. This structure contains detailed information about kernel run such as execution times of individual
kernel functions, status of computation (i.e., if it finished successfully) and more.

```cpp
std::vector<float> output(numberOfElements, 0.0f);

// Add kernel and buffers to tuner
...

const auto result = tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(outputId, output.data())});
```

#### Offline tuning

Todo

#### Online tuning

Todo

----

### Stop conditions

### Searchers

### Utility functions

### Asynchronous execution

### Profiling

### Interoperability

### Python API

### Feature parity across compute APIs
