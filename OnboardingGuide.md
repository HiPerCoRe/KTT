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
* [Kernel arguments](#kernel-arguments)
    * [Scalar arguments](#scalar-arguments)
    * [Vector arguments](#vector-arguments)
    * [Local memory arguments](#local-memory-arguments)
    * [Symbol arguments](#symbol-arguments)

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

TODO consider adding also code for the kernel implementing tuning parameter

```cpp
const size_t numberOfElements = 1024 * 1024;
const ktt::DimensionVector globalDimensions(numberOfElements);
const ktt::DimensionVector localDimensions(64);

const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("kernelName", "kernelFile.cl", globalDimensions, localDimensions);
const ktt::KernelId kernel = tuner.CreateSimpleKernel("TestKernel", definition);

/* Initialize kernel input and output */

tuner.AddParameter(kernel, "unroll_factor", std::vector<uint64_t>{1, 2, 4, 8});
tuner.Tune(kernel);
```

In the code snippet above, we create a kernel definition by specifying the name of kernel function and path to its source file. We also define
its default global and local dimensions (e.g., size of ND-range and work-group in OpenCL, size of grid and block in CUDA). We use provided
kernel definition handle to create kernel. We can also specify custom name for the kernel which is used e.g., for logging purposes. Afterwards,
we can use the kernel handle to define tuning parameter and launch autotuning. The step of creating kernel definition and kernel separately may
seem redundant at first, but it plays important role during more complex use cases that will be covered later.

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

The code above demonstrates how information about all available OpenCL platforms and devices is retrieved from KTT. In this case, the tuner is created
for the first device on the first platform (both platform and device index is 0).

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
separate buffer and thus reduce memory usage.

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

Local (shared in CUDA terminology) memory arguments are used to allocate corresponding amount of cache-like memory which is shared accross all items
(threads) inside a work-group. User has to specify the data type and total size of allocated memory in bytes.

```cpp
// Allocate local memory for 4 floats and 2 integers.
const ktt::ArgumentId local1Id = tuner.AddArgumentLocal<float>(16);
const ktt::ArgumentId local2Id = tuner.AddArgumentLocal<int32_t>(8);
```

#### Symbol arguments

Symbol arguments were added in order to support CUDA arguments marked as `__constant__` or `__device__`. In other APIs, symbol arguments behave in
the same way as scalars since they do not require special handling. In case of CUDA, the name of symbol argument appearing inside CUDA kernel source
code has to be specified during argument addition to tuner.

```cpp
const ktt::ArgumentId symbolId = tuner.AddArgumentSymbol(42, "magicNumber");
```

----

### Creating and running kernel

### Tuning the kernel

### Checking kernel results

### Using kernel launchers

#### Motivation example

#### Launcher implementation

### Using composite kernels

#### Motivation example

#### Composite kernel implementation

### Stop conditions

### Searchers

### Dynamic autotuning

#### Differences over offline tuning

#### Handling kernel arguments

#### Example

### Advanced topics

#### Asynchronous execution

#### Profiling

#### Interoperability

#### Python API
