# Introduction to KTT

When optimizing the performance of compute kernels, a programmer has to make many decisions such as which algorithm to
use, how to arrange data structures in memory, how to block data access to optimize caching or which factor to use for
loop unrolling. Such decisions cannot be typically made in isolation. For example, when the data layout in memory is changed,
a different algorithm may perform better. Therefore, it is necessary to explore a vast amount of combinations of optimization
decisions in order to reach the best performance. Moreover, the best combination of optimization decisions can differ for
various hardware devices or program setup. Therefore, an automatic search for the best combination of these decisions,
called autotuning, is valuable.

Naturally, a batch script can be sufficient for autotuning in a simple use case. However, in advanced applications,
usage of an autotuning framework can be beneficial, as it can automatically handle memory objects, detect errors in autotuned
kernels or perform autotuning during program runtime.

Kernel Tuning Toolkit is a framework that allows autotuning of compute kernels written in CUDA, OpenCL or Vulkan. It provides
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

When leveraging autotuning, a programmer needs to consider which properties of their computation can be autotuned. For
example, an algorithm may contain for loop which can be unrolled. There are multiple options for unroll factor value
of this loop, e.g., 1 (no unroll), 2, 4, 8. Picking the optimal value manually for a particular device is difficult. Therefore
we can define a tuning parameter for the unroll factor with the specified values. Afterward, we can launch four different
versions of our computation to see which value performs best.

In practice, the computations are often complex enough to contain multiple parts that can be optimized, leading
to a definition of many tuning parameters. For example, we may have the previously mentioned loop unroll parameter with
values {1, 2, 4, 8} and another parameter controlling data arrangement in memory with values {0, 1}. Combinations of these
parameters now define eight different versions of computation. One such combination is called tuning configuration. Together, all
tuning configurations define configuration space. The size of the space grows exponentially with the addition of more tuning
parameters. KTT framework offers functionality to mitigate this problem which we will discuss in the follow-up sections.

----

### Simple autotuning example

Offline kernel tuning is the simplest use case of the KTT framework. It involves creating a kernel, specifying its arguments (data),
defining tuning parameters and then launching autotuning. During autotuning, tuning parameter values are propagated to kernel source
code in the form of preprocessor definitions. E.g., when configuration which contains a parameter with name unroll_factor and value 2
is launched, the following code is added at the beginning of kernel source code: `#define unroll_factor 2`. The definitions can be used
to alter kernel functionality based on tuning parameter values.

In the code snippet below, we create a kernel definition by specifying the name of a kernel function and a path to its source file. We also define
its default global and local dimensions (e.g., ND-range and work-group size in OpenCL; grid and block size in CUDA). We use the provided
kernel definition id to create a kernel. We can also specify a custom name for the kernel that is used, e.g., for logging purposes. Afterward,
we can use the kernel id to define a tuning parameter and launch autotuning. The step of separately creating kernel definition and kernel may
seem redundant at first, but it plays a vital role during more complex use cases that we will cover later.

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

The following snippet demonstrates how we could use our previously defined tuning parameter to alter computation inside the kernel.

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

The first step before we can utilize KTT is a creation of a tuner instance. The tuner is one of the major KTT classes and implements a large portion of
autotuning logic. The KTT structures such as kernels, kernel arguments and tuning parameters are tied to a specific tuner instance.
The simplest tuner constructor requires three parameters - index for a platform, index for a device and compute API that will be utilized (e.g., CUDA,
OpenCL). The indices for platforms and devices are assigned by KTT. We can retrieve them through `PlatformInfo` and `DeviceInfo` structures. These
structures also contain some other useful information such as a list of supported extensions, global memory size, a number of available compute units and
more. Note that the assigned indices remain the same when autotuning applications are launched multiple times on the same computer. They only change
when the hardware configuration changes (e.g., a new device is added, an old device is removed, a device driver is reinstalled). Also note, that the indices
may not be the same across multiple compute APIs (e.g., an index for the same device may be different under OpenCL and CUDA).

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

Before launching a kernel via KTT, we must load its source into the tuner. We can achieve this by creating a kernel definition. During its creation,
we specify kernel function name and kernel source. The source can be added either from a string or from a file. Next, we specify default global
(ND-range / grid) and local (work-group / block) sizes. This is done via KTT structure `DimensionVector`, which supports up to three
dimensions. When a kernel is launched during tuning, the thread sizes chosen during kernel definition creation will be used. There are ways to launch
kernels with sizes different from the default, which we will cover later. For CUDA API, the addition of templated kernels is supported as well.
When creating a definition, it is possible to specify types used to instantiate kernel function from a template. When we need to instantiate the same
kernel template with different types, we add multiple kernel definitions with corresponding types, which are handled independently.

Once we have kernel definitions, we can create kernels from them. It is possible to create a simple kernel that only uses one definition and
a composite kernel that uses multiple definitions. Usage of composite kernels is useful for computations that launch multiple kernel
functions in order to compute the result. In this case, it is also necessary to define a kernel launcher which is a function that tells the tuner in which
order and how many times each kernel function is launched. Kernel launchers are covered in detail in a separate section.

Note that KTT terminology regarding kernel definitions and kernels differs slightly from regular compute APIs. KTT kernel definition roughly
corresponds to a single kernel function (also called kernel in e.g., OpenCL or CUDA). KTT kernel corresponds to a specific computation that uses
one or more kernel functions and for which it is possible to define tuning parameters. KTT framework allows sharing of kernel definitions across
multiple kernels (i.e., we can use the same kernel function in multiple computations).

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

Kernel arguments define the input and output of a kernel. KTT supports multiple forms of kernel arguments such as buffers, scalars and constant memory
arguments. The tuner must receive an argument's description before it can be assigned to a kernel. In case of a buffer argument, this includes the
initial data placed inside the buffer before a kernel is launched, its access type (read or write) and the memory location from which kernel accesses the buffer
(host or device). Once the information is provided, the tuner returns a handle to the argument. As the code below shows, we can assign arguments to kernel
definitions through this handle. KTT supports a wide range of data types for kernel arguments, including all built-in integer and floating-point
types as well as custom types. Note, however, that custom types must be trivially copyable, so transferring the arguments into device memory remains possible.

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
change, the buffers must be copied into memory only once before the first kernel configuration is launched and remain the same for subsequent
configurations. Setting correct access types to arguments can therefore lead to better tuning performance.

Next, it is possible to decide the memory location from which a kernel accesses the argument buffer. The two main options are host memory and device
memory. Users may wish to choose a different location depending on the type of device used for autotuning (e.g., host memory for CPUs, device memory for
dedicated GPUs). For host memory, it is additionally possible to use zero-copy optimization. This optimization causes kernels to access the argument data
directly instead of creating a separate buffer and thus reduces memory usage. For CUDA and OpenCL 2.0, one additional memory location option exists - unified.
Unified memory buffers can be accessed from both host and kernel side, relying on a device driver to migrate the data automatically.

Management type option specifies whether buffer management is handled automatically by the tuner (e.g., write arguments are automatically reset
to initial state before a new kernel configuration is launched, buffers are created and deleted automatically) or by the user. In some advanced cases,
users may wish to manage the buffers manually. Note, however, that this requires the usage of kernel launchers which we will discuss later.

The final option for vector arguments is whether the initial data provided by the user should be copied inside the tuner or referenced directly. By default,
the data is copied, which is safer (i.e., temporary arguments work correctly) but less memory efficient. If the initial data is provided in the form of
an lvalue argument, the tuner can use a direct reference to avoid copying. This requires the user to keep the initial data buffer valid while the tuner uses
the argument.

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

Local (shared in CUDA terminology) memory arguments are used to allocate a corresponding amount of cache-like memory, which is shared across all work-items
(threads) inside a work-group (thread block). We just need to specify the data type and total size of allocated memory in bytes.

```cpp
// Allocate local memory for 4 floats and 2 integers.
const ktt::ArgumentId local1Id = tuner.AddArgumentLocal<float>(16);
const ktt::ArgumentId local2Id = tuner.AddArgumentLocal<int32_t>(8);
```

#### Symbol arguments

Symbol arguments were introduced to support CUDA variables marked as `__constant__` or `__device__`. The name of a symbol argument appearing
inside a CUDA kernel source code has to be specified during argument addition to the tuner. Symbol arguments behave the same as scalars in
other APIs since they do not require special handling. In that case, the name of a symbol is ignored.

```cpp
const ktt::ArgumentId symbolId = tuner.AddArgumentSymbol(42, "magicNumber");
```

----

### Tuning parameters

Tuning parameters in KTT can be either unsigned integers or floats. When defining a new parameter, we need to specify its name (i.e., the name through
which it can be referenced in kernel source) and values. With the addition of more tuning parameters, the size of tuning space grows exponentially as we
need to explore all parameter combinations. KTT provides two features for users to slow down the tuning space growth.

```cpp
// We add 4 different parameters, the size of tuning space is 40 (5 * 2 * 4 * 1)
tuner.AddParameter(kernel, "unroll_factor", std::vector<uint64_t>{1, 2, 4, 8, 16});
tuner.AddParameter(kernel, "use_constant_memory", std::vector<uint64_t>{0, 1});
tuner.AddParameter(kernel, "vector_type", std::vector<uint64_t>{1, 2, 4, 8});
tuner.AddParameter(kernel, "float_value", std::vector<double>{1.0});
```

#### Parameter constraints

The first option is tuning constraints. Through constraints, it is possible to tell the tuner to skip generating configurations for certain combinations
of parameters. Parameter constraint is a function that receives values for the specified parameters on input and decides whether that combination is valid.
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
Note that parameter constraints are typically used in three scenarios. First, constraints can remove points in the tuning space (i.e., combinations of tuning parameters' values), which produces invalid code. Consider an example when two-dimensional blocks (work-groups in OpenCL) are created. The constraint can upper-bound thread block size (computed as block's x-dimension multiplied by block's y-dimension), so it does not exceed the highest thread block size executable on GPU. Second, constraints can prune redundant points in tuning space. In the example above, there is no need to tune vector size when the code is not vectorized. Third, constraints can remove points in the tuning space that produce underperforming code. In our example, considering two-dimensional thread blocks, we can constrain tuning space to avoid sub-warp blocks with less than 32 threads.


#### Parameter groups

The second option is tuning parameter groups. This option is mainly helpful for composite kernels with some tuning parameters that only affect one
kernel definition inside the kernel. For example, if we have a composite kernel with two kernel definitions and each definition is affected by three
parameters (we have six parameters in total), and we know that each parameter only affects one specific definition, we can evaluate the two groups
independently. This can significantly reduce the total number of evaluated configurations (e.g., if each of the parameters has two different values,
the total number of configurations is 64 - 2^6; with the usage of parameter groups, it is only 16 - 2^3 + 2^3). It is also possible to combine the use
of constraints and groups. However, constraints can only be added between parameters that belong to the same group.

```cpp
// We add 4 different parameters split into 2 independent groups, reducing size of tuning space from 16 to 8
tuner.AddParameter(kernel, "a1", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "a2", std::vector<uint64_t>{0, 1}, "group_a");
tuner.AddParameter(kernel, "b1", std::vector<uint64_t>{0, 1}, "group_b");
tuner.AddParameter(kernel, "b2", std::vector<uint64_t>{0, 1}, "group_b");
```

#### Thread modifiers

Some tuning parameters can affect the global or local number of threads with which a kernel function is launched. For example, we may have a parameter
that affects the amount of work performed by each thread. The more work each thread does, the fewer (global) threads we need in total to perform computation.
In KTT, we can define such dependency via thread modifiers. The thread modifier is a function that takes a default thread size and changes it based on
values of specified tuning parameters.

When adding a new modifier, we specify a kernel and its definitions whose thread sizes are affected by the modifier. Then we choose whether the modifier
affects the global or local size, its dimension and names of tuning parameters tied to the modifier. The modifier function can be specified through enum,
which supports simple operations such as multiplication or addition, but allows only one tuning parameter to be tied to the modifier. Another
option is using a custom function that can be more complex and supports multiple tuning parameters. Creating multiple thread modifiers for the same thread
type (global/local) and dimension is possible. In that case, the modifiers will be applied in the order of their addition to the tuner. Similar to constraints,
it is possible to tie only integer parameters to thread modifiers.

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

When developing autotuned kernels with a large number of parameters, it is often necessary to check whether each configuration computes the correct output.
KTT provides a way to compare output from tuned kernel configurations with reference output automatically. That means each time a kernel configuration is
finished, the contents of its output buffer are transferred into host memory and then compared to precomputed reference output. It is possible to compute
the reference in two ways.

#### Reference computation

Reference computation is a function that computes the reference output in host code and stores the result in the buffer provided by KTT. The size of
that buffer matches the size of the validated kernel output buffer. When defining a reference computation, we only need to provide the function and
the validated output argument's id.

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

Another option is to compute a reference result with a kernel. We need to provide the reference kernel's id and the validated output argument's id
in this case. A reference kernel can have tuning parameters as well, so there is an option to choose a specific reference configuration. We can
provide an empty configuration if a reference kernel has no parameters. The reference kernel may be the same as the tuned kernel (e.g., using some
default configuration known to work).

```cpp
tuner.SetReferenceKernel(outputId, referenceKernel, ktt::KernelConfiguration());
```

#### Validation customization

There are certain ways to customize further how the tuner performs validation. By default, the entire output buffer is validated. If validating only
a portion of the buffer is sufficient, setting a custom validation range is possible. In this case, the size of the reference buffer provided by KTT
for reference computation validation will be automatically adjusted as well.

Validation works out-of-the-box for integer and floating-point argument data types. In the case of floating-point arguments, it is possible to choose
validation method (e.g., comparing each element separately or summing up all elements and comparing the result) and tolerance threshold since
different kernel configurations may have varying accuracy of computing floating-point output.

If arguments with user-defined types are validated, it is necessary to define a value comparator. A comparator is a function that receives two
elements with the specified type on input and decides whether they are equal. A custom comparator can optionally be used for integer and floating-point
data types as well, to override the default comparison functionality.

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

Kernel launchers enable users to customize how kernels are run inside KTT. Launcher is a function that defines what happens when a kernel under
a particular configuration is launched via the tuner. For simple kernels, a default launcher is provided by KTT. This launcher runs the kernel
function tied to the kernel and waits until it has finished. If a computation requires launching a kernel function multiple times, running some
part in host code or using multiple kernel functions, we need to define our own launcher. In the case of composite kernels, defining a custom
launcher is mandatory since KTT does not know the order in which it should run the individual kernel functions.

Kernel launcher has access to a low-level KTT compute interface on input. This interface makes it possible to launch kernel functions, change
their thread sizes, modify buffers and retrieve the current kernel configuration. This enables tuning parameters to affect computation
behavior in host code in addition to modifying kernel behavior. The modifications to kernel arguments and buffers inside a launcher are isolated
to the specific kernel configuration launch. Therefore, it is not necessary to reset arguments to their original values for each kernel launch; it is
done automatically by the tuner. The only exception to this is the usage of user-managed vector arguments; those have to be reset manually.

```cpp
// This launcher is equivalent in functionality to the default simple kernel launcher provided by KTT.
tuner.SetLauncher(kernel, [definition](ktt::ComputeInterface& interface)
{
    interface.RunKernel(definition);
});
```

----

### Kernel running and tuning modes

KTT supports kernel tuning as well as standard kernel running. Running kernels via tuner is often more convenient than directly using a specific
compute API since a lot of boilerplate code such as compute queue management and kernel source compilation is abstracted. It is possible to specify
a configuration under which the kernel is run, so the workflow where a kernel is first tuned and then launched repeatedly with the best configuration is
supported. It is possible to transfer kernel output into host memory by utilizing the `BufferOutputDescriptor` structure. When creating this structure,
we need to specify the id of a buffer that should be transferred and a pointer to memory where the buffer contents should be saved. It is possible to pass
multiple such structures into the kernel running method - each descriptor corresponds to a single buffer that should be transferred. After a kernel run is
finished, the `KernelResult` structure is returned. This structure contains detailed information about the run, such as execution times of individual
kernel functions, the status of computation (i.e., if it finished successfully) and more.

```cpp
std::vector<float> output(numberOfElements, 0.0f);

// Add kernel and buffers to the tuner
...

const auto result = tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(outputId, output.data())});
```

#### Offline tuning

During offline tuning, kernel configurations are run one after another without user interference. Therefore, this mode separates finding the best
configuration and subsequent usage of the tuned kernel in an application. This enables the tuner to implement some optimizations that would otherwise not be
possible, for example, caching of read-only buffers over multiple kernel runs under different configurations. By default, the entire configuration space is
explored during offline tuning. We can alter this by leveraging stop conditions described in the next section.

Kernel output cannot be retrieved during offline tuning because all configurations are launched within a single API call. After the tuning ends, the tuner
returns the list of `KernelResult` structures corresponding to all tested configurations. We can save these results either in XML or JSON format for
further analysis.

```cpp
const std::vector<ktt::KernelResult> results = tuner.Tune(kernel);
tuner.SaveResults(results, "TuningOutput", ktt::OutputFormat::JSON);
```

#### Online tuning

Online tuning combines kernel tuning with regular running. We can retrieve and use the output from each kernel run like during kernel running. However,
we do not specify the configuration under which kernel is run, but the tuner launches a different configuration each time a kernel is launched, similar to
offline tuning. This mode does not separate tuning and usage of a tuned kernel but enables both to happen simultaneously. This can be beneficial
in situations where offline tuning is impractical (e.g., when the size of kernel input is frequently changed, which causes the optimal configuration
to change as well). If a kernel is launched via online tuning after exploring all configurations, the best configuration is used.

```cpp
std::vector<float> output(numberOfElements, 0.0f);

// Add kernel and buffers to the tuner
...

const auto result = tuner.TuneIteration(kernel, {ktt::BufferOutputDescriptor(outputId, output.data())});
```

#### Accuracy of tuning results

In order to identify the best configuration accurately, it is necessary to launch all configurations under the same conditions so that metrics such as
kernel function execution times can be objectively compared. This means that tuned kernels should be launched on the target device in isolation.
Launching multiple kernels concurrently while performing tuning may cause inaccuracies in collected data. Furthermore, if the size of kernel input is
changed (e.g., during online tuning), we should restart the tuning process from the beginning since the input size often affects the best configuration.
We can achieve the restart by calling the `ClearData` API method.

----

### Stop conditions

We can utilize stop conditions to interrupt offline tuning when certain criteria are met. The stop condition is initialized before offline tuning begins
and updated after each tested configuration. Within the update, the condition has access to the `KernelResult` structure from prior kernel run. KTT currently
offers the following stop conditions:
* ConfigurationCount - tuning stops after reaching the specified number of tested configurations.
* ConfigurationDuration - tuning stops after a configuration with execution time below the specified threshold is found.
* ConfigurationFraction - tuning stops after exploring the specified fraction of configuration space.
* TuningDuration - tuning stops after the specified duration has passed.

The stop condition API is public, allowing users to create their own stop conditions. All of the built-in conditions are implemented in public API, so
it is possible to modify them as well.

----

### Searchers

Searchers decide the order in which kernel configurations are selected and run during offline and online tuning. Having an efficient searcher can significantly
reduce the time it takes to find a well-performing configuration. Like stop conditions, a searcher is initialized before tuning begins and updated after
each tested configuration with access to the `KernelResult` structure from the previous run. Searchers are assigned to kernels individually so that each kernel
can have a different searcher. The following searchers are available in KTT API:
* DeterministicSearcher - always explores configurations in the same order (provided that tuning parameters, order of their addition and their values were not changed).
* RandomSearcher - explores configurations in random order.
* McmcSearcher - utilizes Markov chain Monte Carlo method to predict well-performing configurations more accurately than random searcher.

The searcher API is public so that users can implement their own searchers. The API also includes utility methods to simplify custom searcher
implementation. These include a method to get random unexplored configuration or neighboring configurations (configurations that differ in a small
number of parameter values compared to the specified configuration).

----

### Utility functions

KTT provides many utility functions to customize tuner behavior further. The following list contains descriptions of certain functions which can be handy
to use:
* `SetCompilerOptions` - sets options for kernel source code compiler used by compute API (e.g., NVRTC for CUDA).
* `SetGlobalSizeType` - compute APIs use different ways for specifying global thread size (e.g., grid size or ND-range size). This method makes it possible
to override the global thread size format to the one used by the specified API. Usage of this method makes it easier to port programs between different compute
APIs.
* `SetAutomaticGlobalSizeCorrection` - tuner automatically ensures that global thread size is divisible by local thread size. This is required by certain compute
APIs such as OpenCL.
* `SetKernelCacheCapacity` - changes size of a cache for compiled kernels. KTT utilizes the cache to improve performance when the same kernel function with the
same configuration is launched multiple times (e.g., inside kernel launcher or during kernel running).
* `SetLoggingLevel` - controls the amount of logging information printed to the output. Higher levels print more detailed information which aids debugging.
* `SetTimeUnit` - specifies time unit used for printing execution times. This affects console output as well as kernel results saved into a file.

----

### Profiling metrics collection

Apart from execution times, KTT can also collect other types of information from kernel runs. This includes low-level profiling metrics from kernel function
executions such as global memory utilization, number of executed instructions and more. These metrics can be utilized e.g., by searchers to find well-performing
configurations faster. The collection of profiling metrics is disabled by default as it changes the default tuning behavior. In order to collect all profiling
metrics, it is usually necessary to run the same kernel function multiple times (the number increases when more metrics are collected). It furthermore requires
kernels to be run synchronously. Enabling profiling metrics collection thus decreases tuning performance. It is possible to mitigate performance impact by allowing
only specific metrics, which can be done through KTT API.

Collection of profiling metrics is currently supported for Nvidia devices on CUDA backend and AMD devices on OpenCL backend. Intel devices are unsupported
at the moment due to a lack of profiling library support. Profiling metrics can also be collected for composite kernels. Note, however, that the metrics
collection is restricted to a single definition within a composite kernel for AMD devices and newer Nvidia devices (Turing and onwards). This is due to profiling
library limitations.

#### Interaction with online tuning and kernel running

When utilizing kernel running and online tuning, it is possible to decrease further the performance impact of executing the same kernel function multiple times
during profiling. Rather than performing all of the profiling runs at once, it is possible to split the profiling metric collection over multiple online tuning or
kernel running API function invocations and utilize output from each run. The intermediate `KernelResult` structures from such runs will not contain valid profiling
metrics, but the other data will remain accurate. Once the profiling for the current configuration is concluded, the final kernel result will have valid
profiling data.

----

### Interoperability

The KTT framework could originally be used only in isolation to create standalone programs focused on tuning a specific kernel. In recent versions, the API
was extended to support tuner integration into larger software suites. Multiple major features contribute to this support. They are described in this section.

#### Custom compute library initialization

By default, when the tuner is created, it initializes its own internal compute API structures such as context, compute queues and buffers. However, it is also
possible to use the tuner with custom structures. This enables tuner integration into libraries that perform their own compute API initialization.
During tuner initialization, we can pass the `ComputeApiInitializer` structure to it. This structure contains our own context and compute queues. When adding
a vector argument, it is possible to pass our own compute buffer, which the tuner will then utilize. These structures remain under our own management; the tuner
will just reference them and use them when needed. Before releasing these structures, the tuner should be destroyed first so that it can perform a proper cleanup.
Note, however, that the tuner will never release the referenced structures on its own.

#### Asynchronous execution

All kernel function runs and buffer data transfers are synchronized when performing tuning. This is necessary to obtain accurate tuning data. Applications that
combine kernel tuning and kernel running can enable asynchronous kernel launches and buffer transfers after tuning is completed. This is achieved by utilizing
kernel launchers and compute interface. The compute interface API contains methods for asynchronous operations. They enable us to choose a compute queue for
launching an operation and return event id, which can be later used to wait for the operation to complete. Note, however, that kernel results returned from
asynchronous launches will contain inaccurate execution timings since the results may be returned before the asynchronous operation has finished. Therefore,
the asynchronous execution should be utilized only for kernel running, not tuning.

#### Lifetime of internal tuner structures

Internal KTT structures such as kernels, kernel definitions, arguments and configuration data have their lifetimes tied to the tuner. Some applications which
utilize the tuner may prefer to remove these structures on the fly to save memory. Currently, it is possible to remove kernels, kernel definitions, arguments and
user-provided compute queues from the tuner by specifying their ids. When removing a kernel, all of its associated data such as generated configurations, parameters
and validation data are also removed. Note that it is not possible to remove structures referenced by other structures. E.g., when removing a kernel definition, we
must first remove all kernels which utilize that definition.

----

### Python API

The native KTT API is available in C++. Users who prefer Python have an option to build KTT as a Python module which can then be imported into Python. The majority
of KTT API methods can be afterward called directly from Python while still benefitting from the performance of the KTT module built in C++. It is also possible to
implement custom searchers and stop conditions directly in Python. Therefore, users can take advantage of libraries available in Python but not in C++ for more
complex searcher implementations. The majority of functions, enums and classes have the same names and arguments as in C++. A small number of limitations is
described in the follow-up subsection.

#### Python limitations

Almost the entire KTT API is available in Python. However, certain features are restricted to C++ API due to limitations in Python language and utilized
libraries. They are the following:
* Templated methods - Python does not support templates, so there are separate versions of methods for different data types instead (e.g., `AddArgumentVectorFloat`,
`AddArgumentVectorInt`). The addition of kernel arguments with custom types is not supported either.
* Custom library initialization - Custom context, compute queues and buffers cannot be used in Python.
* Methods that use void pointers in C++ API - Python does not have a direct equivalent to void* type. It is necessary to utilize a low-level `ctypes` Python
module to interact with these methods through `PyCapsule` objects.

----

### Feature parity across compute APIs

KTT framework aims to maintain feature parity across all of its supported compute APIs (OpenCL, CUDA and Vulkan). That means if a particular feature is supported in
the KTT CUDA backend, it should also be available in OpenCL and Vulkan backends, provided that it is natively supported in those APIs. There are some exceptions
to that:
* Vulkan backend limitations - certain features are currently unsupported in Vulkan due to development time constraints. These include support for profiling metrics,
unified and zero-copy buffers and a subset of advanced buffer handling methods. The support for these features may still be added at a later time.
* Unified memory in OpenCL - the usage of unified OpenCL buffers requires OpenCL 2.0. Some devices (e.g., Nvidia GPUs) still do not support this OpenCL version.
* Templated kernel functions - templates are currently limited to CUDA kernels due to lack of support in other APIs.
