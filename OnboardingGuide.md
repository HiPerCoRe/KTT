# Introduction to KTT

When optimizing performance of compute kernels, a programmer has to make a lot of decisions such as which algorithm to
use, how to arrange data structures in memory, how to block data access to optimize caching or which factor to use for
loop unrolling. Such decisions cannot be typically made in isolation - for example, when data layout in memory is changed,
a different algorithm may perform better. Therefore, it is necessary to explore vast amount of combinations of optimization
decisions in order to reach the best performance. Moreover, the best combination of optimization decisions can differ for
various hardware devices or program setup. Therefore, a way of automatic search for the best combination of these decisions,
called autotuning, is valuable.

Naturally, in the simple use case, a batch script can be sufficient for autotuning. However, in advanced applications,
usage of an autotuning framework can be beneficial, as it can automatically handle memory objects, detect errors in autotuned
kernels or perform autotuning during program runtime.

Kernel Tuning Toolkit is a framework which allows autotuning of compute kernels written in CUDA, OpenCL or Vulkan. It provides
unified interface for those APIs, handles communication between host (CPU) and accelerator (GPU, Xeon Phi, etc.), checks results
and timing of tuned kernels, allows dynamic (online) tuning during program runtime, profiling of autotuned kernels and more.

----

### Table of contents
* [Basic principles behind KTT](#basic-principles-behind-ktt)
* [Simple autotuning example](#simple-autotuning-example)
* [KTT initialization](#ktt-initialization)
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
    * [Accuracy of tuning results](#accuracy-of-tuning-results)
* [Stop conditions](#stop-conditions)
* [Searchers](#searchers)
* [Utility functions](#utility-functions)
* [Profiling metrics collection](#profiling-metrics-collection)
    * [Interaction with online tuning and kernel running](#interaction-with-online-tuning-and-kernel-running)
* [Interoperability](#interoperability)
    * [Custom compute library initialization](#custom-compute-library-initialization)
    * [Asynchronous execution](#asynchronous-execution)
    * [Lifetime of internal tuner structures](#lifetime-of-internal-tuner-structures)
* [Python API](#python-api)
    * [Python limitations](#python-limitations)
* [Feature parity across compute APIs](#feature-parity-across-compute-apis)

----

### Basic principles behind KTT

When leveraging autotuning, a programmer needs to think about which properties of their computation can be autotuned. For
example, an algorithm may contain for loop which can be unrolled. There are multiple options for unroll factor value
of this loop, e.g., 1 (no unroll), 2, 4, 8. Picking the optimal value for a certain device manually is difficult, therefore
we can define a tuning parameter for the unroll factor with the specified values. Afterwards, we can launch four different
versions of our computation to see which value performs best.

In practice, the computations are often complex enough to contain multiple parts which can be optimized in this way, leading
to definition of multiple tuning parameters. For example we may have the previously mentioned loop unroll parameter with
values {1, 2, 4, 8} and another parameter controlling data arrangement in memory with values {0, 1}. Combinations of these
parameters now define 8 different versions of computation. One such combination is called tuning configuration. Together, all
tuning configurations define configuration space. The size of the space grows exponentially with addition of more tuning
parameters. KTT framework offers functionality to mitigate this problem which will be discussed in the follow-up sections.

----

### Simple autotuning example

Offline kernel tuning is the simplest use case of KTT framework. It involves creating a kernel, specifying its arguments (data),
defining tuning parameters and then launching autotuning. During autotuning, tuning parameter values are propagated to kernel source
code in a form of preprocessor definitions. E.g., when configuration which contains parameter with name unroll_factor and value 2
is launched, the following code is added at the beginning of kernel source code: `#define unroll_factor 2`. The definitions can be used
to alter kernel functionality based on tuning parameter values.

In the code snippet below, we create a kernel definition by specifying the name of kernel function and path to its source file. We also define
its default global and local dimensions (e.g., size of ND-range and work-group in OpenCL, size of grid and block in CUDA). We use the provided
kernel definition id to create kernel. We can also specify custom name for the kernel which is used e.g., for logging purposes. Afterwards,
we can use the kernel id to define a tuning parameter and launch autotuning. The step of creating kernel definition and kernel separately may
seem redundant at first but it plays an important role during more complex use cases that will be covered later.

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

### KTT initialization

The first step before we can utilize KTT is creation of a tuner instance. Tuner is one of the major KTT classes and implements large portion of
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

Before kernel can be launched via KTT, its source must be loaded into tuner. This is achieved by creating a kernel definition. During its creation,
we specify kernel function name and kernel source. The source can be added either from string or from file. Next, we specify default global
(NDrange / grid) and local (work-group / block) sizes. The sizes are specified with KTT structure `DimensionVector` which supports up to three
dimensions. When a kernel is launched during tuning, the thread sizes chosen during kernel definition creation will be used. There are ways to launch
kernels with different than default sizes which will be covered later. For CUDA API, addition of templated kernels is supported as well. When creating
a definition, it is possible to specify types that should be used to instantiate kernel function from template. When we need to instantiate the same
kernel template with different types, we do that by adding multiple kernel definitions with corresponding types which are then handled independently.

Once we have kernel definitions, we can create kernels from them. It is possible to create a simple kernel which only uses one definition as well as
a composite kernel which uses multiple definitions. Usage of composite kernels is useful for computations which require launching of multiple kernel
functions in order to compute the result. In this case it is also necessary to define kernel launcher which is a function that tells the tuner in which
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
types as well as custom types. Note however, that custom types must be trivially copyable, so it remains possible to transfer the arguments into device memory.

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

Vector arguments have more customization options available than scalars. Other than the initial data, it is possible to specify whether an argument
is used for reading or writing. For read-only arguments, additional optimization is possible during offline tuning. Since their contents do not
change, the buffers need to be copied into memory only once before the first kernel configuration is launched and then remain the same for subsequent
configurations. Setting correct access types to arguments can therefore lead to better tuning performance.

Next, it is possible to decide memory location from which the argument buffer is accessed by kernel - the two main options are host memory and device
memory. Users may wish to choose different location depending on the type of device used for autotuning (e.g., host memory for CPUs, device memory for
dedicated GPUs). For host memory, it is additionally possible to use zero-copy optimization. This optimization causes kernels to access the argument data
directly, instead of creating a separate buffer and thus reduces memory usage. For CUDA and OpenCL 2.0, one additional memory location option exists - unified.
Unified memory buffers can be accessed from both host and kernel side, relying on device driver to take care of migrating the data automatically.

Management type option specifies whether buffer management is handled automatically by the tuner (e.g., write arguments are automatically reset
to initial state before new kernel configuration is launched, buffers are created and deleted automatically) or by the user. In some advanced cases,
users may wish to manage the buffers manually. Note however, that this requires usage of kernel launchers which will be discussed later.

The final option for vector arguments is whether the initial data provided by user should be copied inside the tuner or referenced directly. By default,
the data is copied which is safer (i.e., temporary arguments work correctly) but less memory efficient. In case the initial data is provided in form of
lvalue argument, direct reference can be used to avoid copying. This requires user to keep the initial data buffer valid during time the argument is
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
(threads) inside a work-group (thread block). We just need to specify the data type and total size of allocated memory in bytes.

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

Tuning parameters in KTT can be either unsigned integers or floats. When defining a new parameter, we need to specify its name (i.e., the name through
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
of parameters. Constraint is a function which receives values for the specified parameters on input and decides whether that combination is valid.
We can choose which parameters are evaluated by a specific constraint. Note that currently, it is possible to add constraints only between integer
parameters.

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
kernel definition inside the kernel. For example, if we have a composite kernel with two kernel definitions and each definition is affected by three
parameters (we have six parameters in total), and we know that each parameter only affects one specific definition, we can evaluate the two parameter
groups independently. This can greatly reduce the total number of evaluated configurations (e.g., if each of the parameters has two different values,
the total number of configurations is 64 - 2^6; with usage of parameter groups, it is only 16 - 2^3 + 2^3). It is also possible to combine usage of
constraints and groups, however constraints can only be added between parameters which belong into the same group.

```cpp
// We add 4 different parameters split into 2 independent groups, reducing size of tuning space from 16 to 8
tuner.AddParameter(kernel, "a1", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "a2", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "b1", std::vector<uint64_t>{0, 1}, "group_b");
tuner.AddParameter(kernel, "b2", std::vector<uint64_t>{0, 1}, "group_b");
```

#### Thread modifiers

Some tuning parameters can affect global or local number of threads a kernel function is launched with. For example, we may have a parameter which
affects amount of work performed by each thread. The more work each thread does, the less (global) threads we need in total to perform computation.
In KTT, we can define such dependency via thread modifiers. The thread modifier is a function which takes a default thread size and changes it based on
values of specified tuning parameters.

When adding a new modifier, we specify kernel and its definitions whose thread sizes are affected by the modifier. Then we choose whether modifier
affects global or local size, its dimension and names of tuning parameters tied to modifier. The modifier function can be specified through enum
which supports certain simple functions such as multiplication or addition, but allows only one tuning parameter to be tied to modifier. Another
option is using a custom function which can be more complex and supports multiple tuning parameters. It is possible to create multiple thread
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

When developing autotuned kernels with large number of parameters, it is often necessary to check whether each configuration computes the correct output.
KTT provides a way to automatically compare output from tuned kernel configurations to reference output. That means each time a kernel configuration is
finished, the contents of its output buffer are transferred into host memory and then compared to precomputed reference output. The reference can be
computed in two ways.

#### Reference computation

Reference computation is a function which computes the reference output in host code and stores the result in the buffer provided by KTT. The size of
that buffer matches the size of validated kernel output buffer. When defining a reference computation, we only need to provide the function and the id
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
argument. It is possible for reference kernel to have tuning parameters as well, so there is an option to choose a specific reference configuration. If
a reference kernel has no parameters, empty configuration can be provided. The reference kernel may be the same as tuned kernel (e.g. using some default
configuration that is known to work).

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

If arguments with user-defined types are validated, it is necessary to define a value comparator. Comparator is a function which receives two
elements with the specified type on input and decides whether they are equal. A custom comparator can optionally be used for integer and floating-point
data types as well, in order to override the default comparison functionality.

```cpp
struct KernelData
{
    float a;
    float b;
    float result;
};

tuner.SetValueComparator(dataId, [](const void* resultPointer, const void* referencePointer)
{
    const auto* result = static_cast<const KernelData*>(resultPointer);
    const auto* reference = static_cast<const KernelData*>(referencePointer);

    if (result->result != reference->result)
    {
        std::cerr << "Result " << result->result << " does not equal reference " << reference->result << std::endl;
        return false;
    }

    return true;
});
```

----

### Kernel launchers

Kernel launchers enable users to customize how kernels are run inside KTT. Launcher is a function which defines what happens when kernel under
certain configuration is launched via tuner. For simple kernels, a default launcher is provided by KTT. This launcher simply runs the kernel
function tied to kernel and waits until it has finished. If a computation requires launching a kernel function multiple times, running some
part in host code or using multiple kernel functions, then we need to define our own launcher. In case of composite kernels, defining a custom
launcher is mandatory, since KTT does not know the order in which the individual kernel functions should be run.

Kernel launcher has access to low-level KTT compute interface on input. Through this interface, it is possible to launch kernel functions, change
their thread sizes, modify buffers and retrieve the current kernel configuration. This makes it possible for tuning parameters to affect computation
behaviour in host code in addition to modifying kernel behavior. The modifications done to kernel arguments and buffers inside a launcher are isolated
to the specific kernel configuration launch. Therefore, it is not necessary to reset arguments to their original values for each kernel launch, it is
done automatically by the tuner. The only exception to this is usage of user-managed vector arguments, those have to be reset manually.

```cpp
// This launcher is equivalent in functionality to the default simple kernel launcher provided by KTT.
tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
{
    interface.RunKernel(definition);
});
```

----

### Kernel running and tuning modes

KTT supports kernel tuning as well as ordinary kernel running. Running kernels via tuner is often more convenient compared to directly using specific
compute API, since a lot of boilerplate code such as compute queue management and kernel source compilation is abstracted. It is possible to specify
configuration under which the kernel is run, so the workflow where kernel is first tuned and then launched repeatedly with the best configuration, is
supported. It is possible to transfer kernel output into host memory by utilizing `BufferOutputDescriptor` structure. When creating this structure,
we need to specify id of a buffer that should be transferred and pointer to memory where the buffer contents should be saved. It is possible to pass
multiple such structures into kernel running method - each structure corresponds to a single buffer that should be transferred. After a kernel run is
finished, `KernelResult` structure is returned. This structure contains detailed information about the run such as execution times of individual
kernel functions, status of computation (i.e., if it finished successfully) and more.

```cpp
std::vector<float> output(numberOfElements, 0.0f);

// Add kernel and buffers to tuner
...

const auto result = tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(outputId, output.data())});
```

#### Offline tuning

During offline tuning, kernel configurations are run one after another without user interference. This mode therefore separates finding the best
configuration and subsequent usage of tuned kernel in an application. This enables tuner to implement certain optimizations which would otherwise not be
possible, for example caching of read-only buffers over multiple kernel runs under different configurations. By default, the entire configuration space is
explored during offline tuning. This can be altered by leveraging stop conditions, which are described in the next section.

Kernel output cannot be retrieved during offline tuning because all of the configurations are launched within a single API call. The list of `KernelResult`
structures corresponding to all tested configurations is returned after the tuning ends. These results can be saved either in XML or JSON format for
further analysis.

```cpp
const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);
tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);
```

#### Online tuning

Online tuning combines kernel tuning with regular running. Similar to kernel running, we can retrieve and use output from each kernel run. However, we
do not specify the configuration under which kernel is run, but tuner launches a different configuration each time a kernel is launched, similar to
offline tuning. This mode does not separate tuning and usage of a tuned kernel, but rather enables both to happen simultaneously. This can be beneficial
in situations where employment of offline tuning is impractical (e.g., when the size of kernel input is frequently changed which causes the optimal
configuration to change as well). If a kernel is launched via online tuning after all configurations were already explored, the best configuration is used.

```cpp
std::vector<float> output(numberOfElements, 0.0f);

// Add kernel and buffers to tuner
...

const auto result = tuner.TuneIteration(kernel, {ktt::BufferOutputDescriptor(outputId, output.data())});
```

#### Accuracy of tuning results

In order to identify the best configuration accurately, it is necessary to launch all configurations under the same conditions so that metrics such as
kernel function execution times can be objectively compared. This means that tuned kernels should be launched on the target device in isolation.
Launching multiple kernels concurrently while tuning is performed may cause inaccuracies in collected data. Furthemore, if the size of kernel input is
changed (e.g., during online tuning), the tuning process should be restarted from the beginning, since the size of input often affects the best configuration.
The restart can be achieved by calling `ClearData` API method.

----

### Stop conditions

Stop conditions can be used to stop offline tuning when certain criteria is met. The stop condition is initialized before offline tuning begins and updated
after each tested configuration. Within the update, it has access to `KernelResult` structure from prior kernel run. It can utilize this data to check or update
its criteria. KTT currently offers the following stop conditions:
* ConfigurationCount - tuning stops after reaching the specified number of tested configurations.
* ConfigurationDuration - tuning stops after a configuration with execution time below the specified threshold is found.
* ConfigurationFraction - tuning stops after exploring the specified fraction of configuration space.
* TuningDuration - tuning stops after the specified duration has passed.

The stop condition API is public, which means that users can also create their own stop conditions. All of the built-in conditions are implemented in public
API, so it possible to modify them as well.

----

### Searchers

Searchers decide the order in which kernel configurations are selected and run during offline and online tuning. Having an efficient searcher can significantly
reduce the time it takes to find well-performing configuration. Similar to stop conditions, a searcher is initialized before tuning begins and is updated after
each tested configuration with access to `KernelResult` structure from the previous run. Searchers are assigned to kernels individually, so each kernel can have
a different seacher. The following searchers are available in KTT API:
* DeterministicSearcher - always explores configurations in the same order (provided that tuning parameters, order of their addition and their values were not changed).
* RandomSearcher - explores configurations in random order.
* McmcSearcher - utilizes Markov chain Monte Carlo method to predict well-performing configurations more accurately than random searcher.

The searcher API is public, so users can implement their own searchers. The API also includes certain common utility methods to make the custom searcher
implementation easier. These include a method to get random unexplored configuration or neighbouring configurations (configurations which differ in a small
number of parameter values compared to the specified configuration).

----

### Utility functions

KTT provides many utility functions to further customize tuner behavior. The following list contains descriptions of certain functions which can be handy
to use:
* `SetCompilerOptions` - sets options for kernel source code compiler used by compute API (e.g., NVRTC for CUDA).
* `SetGlobalSizeType` - compute APIs use different ways for specifying global thread size (e.g., grid size or ND-range size). This method makes it possible
to override the global thread size format to the one used by the specified API. Usage of this method makes it easier to port programs between different compute
APIs.
* `SetAutomaticGlobalSizeCorrection` - tuner automatically ensures that global thread size is divisible by local thread size. This is required by certain compute
APIs such as OpenCL.
* `SetKernelCacheCapacity` - changes size of a cache for compiled kernels. KTT utilizes the cache to improve performance when the same kernel function with the
same configuration is launched multiple times (e.g., inside kernel launcher or during kernel running).
* `SetLoggingLevel` - controls the amount of logging information printed to output. Higher levels print more detailed information which is useful for debugging.
* `SetTimeUnit` - specifies time unit used for printing execution times. Affects console output as well as kernel results saved into a file.

----

### Profiling metrics collection

Apart from execution times, KTT can also collect other types of information from kernel runs. This includes low-level profiling metrics from kernel function
executions such as global memory utilization, number of executed instructions and more. These metrics can be utilized e.g., by searchers to find well-performing
configurations faster. The collection of profiling metrics is disabled by default as it changes the default tuning behaviour. In order to collect all profiling
metrics, it is usually necessary to run the same kernel function multiple times (the number increases when more metrics are collected). It furthemore requires
kernels to be run synchronously. Enabling profiling metrics collection thus decreases tuning performance. It is possible to mitigate performance impact by enabling
only certain metrics, which can be done through KTT API.

Collection of profiling metrics is currently supported for Nvidia devices on CUDA backend and AMD devices on OpenCL backend. Intel devices are currently unsupported
due to lack of profiling library support. Profiling metrics can also be collected for composite kernels. Note however, that for AMD devices and newer Nvidia devices
(Turing and onwards), collection of metrics is restricted to a single kernel definition within a composite kernel due to profiling library limitations.

#### Interaction with online tuning and kernel running

When utilizing kernel running and online tuning, it is possible to further decrease performance impact of having to execute the same kernel function multiple times
during profiling. Rather than performing all of the profiling runs at once, it is possible to split the profiling metric collection over multiple online tuning or
kernel running API function invocations and utilize output from each run. The intermediate `KernelResult` structures from such runs will not contain valid profiling
metrics, but still have the remaining data accurate. Once the profiling for the current configuration is concluded, the final kernel result will contain valid
profiling data.

----

### Interoperability

The KTT framework could originally be used only in isolation to create standalone programs which are focused on tuning a specific kernel. In recent versions, the API
was extended to also support tuner integration into larger software suites. There are multiple major features which contribute to this support. They are described
in this section.

#### Custom compute library initialization

By default, when tuner is created, it initializes its own internal compute API structures such as context, compute queues and buffers. It is however possible to
also use the tuner with custom structures as well. This enables tuner integration into libraries which need to perform their own compute API initialization.
During tuner initialization, we can pass `ComputeApiInitializer` structure to it. This structure contains our own context and compute queues. When adding a vector
argument, it is possible to pass our own compute buffer which will then be utilized by tuner. All of these structures still remain under our own management, tuner
will simply reference them and use them when needed. Before releasing these structures, the tuner should be destroyed first, so it can perform proper cleanup. Note
however, that the tuner will never destroy the referenced structures on its own.

#### Asynchronous execution

When performing tuning, all kernel function runs and buffer data transfers are synchronized. This is necessary to obtain accurate tuning data. Applications which
combine kernel tuning and kernel running have an option to enable asynchronous kernel launches and buffer transfers after tuning is completed. This can be achieved
by utilizing kernel launchers and compute interface. The compute interface API contains methods for asynchronous operations. They enable us to choose a compute
queue for launching an operation and return event id which can be later used to wait for the operation to complete. Note however, that kernel results returned from
asynchronous launches will contain inaccurate execution times, since the results may be returned before the asynchronous operation has finished. This feature should
therefore be utilized only for kernel running, not tuning.

#### Lifetime of internal tuner structures

Internal KTT structures such as kernels, kernel definitions, arguments and configuration data have their lifetimes tied to tuner. Certain applications which utilize
tuner may prefer to remove some of these structures on-the-fly to save memory. Currently, it is possible to remove kernels, kernel definitions, arguments and
user-provided compute queues from the tuner by specifying their ids. When removing a kernel, all of its associated data such as generated configurations, parameters
and validation data are removed as well. Note that it is not possible to remove structures which are referenced by other structures. E.g., when removing a kernel
definition, we must make sure that all kernels which utilize that definition are removed first.

----

### Python API

The native KTT API is available in C++. Users who prefer Python have an option to build KTT as Python module which can be then imported into Python. The majority of
KTT API methods can be afterwards called directly from Python while still benefitting from perfomance of KTT module built in C++. It is also possible to implement
custom searchers and stop conditions directly in Python. Users can therefore take advantage of certain libraries available in Python but not in C++ for more
complex searcher implementations. Majority of functions, enums and classes have the same names and arguments as in C++. A small number of limitations is described
in the follow-up subsection.

#### Python limitations

Almost the entire KTT API is available in Python. There are however certain features which are restricuted to C++ API due to limitations in Python language and
utilized libraries. They are the following:
* Templated methods - Python does not support templates, so there are separate versions of methods for different data types instead (e.g., `AddArgumentVectorFloat`,
`AddArgumentVectorInt`). Addition of kernel arguments with custom types is also not supported.
* Custom library initialization - Custom context, compute queues and buffers cannot be used in Python.
* Methods which use void pointers in C++ API - Python does not have a direct equivalent to void* type. It is necessary to utilize low-level `ctypes` Python
module to be able to interact with these methods through `PyCapsule` objects.

----

### Feature parity across compute APIs

KTT framework aims to maintain feature parity across all of its supported compute APIs (OpenCL, CUDA and Vulkan). That means if a certain feature is supported in
KTT CUDA backend, it should also be available in OpenCL and Vulkan backends, provided that the feature is natively supported in those APIs. There are certain
exceptions to that:
* Vulkan backend limitations - certain features are currently unsupported in Vulkan due to development time constraints. These include support for profiling metrics,
unified and zero-copy buffers and certain advanced buffer handling methods. The support for these features may still be added at a later time.
* Unified memory in OpenCL - usage of unified OpenCL buffers requires support for OpenCL 2.0. Certain devices (e.g., Nvidia GPUs) still have this support unfinished.
* Templated kernel functions - templates are currently limited to CUDA kernels due to lack of support in other APIs.
