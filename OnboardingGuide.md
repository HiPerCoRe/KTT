# Introduction

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
    * [Subsection](#subsection)

----

### Basic principles behind KTT

When leveraging autotuning, a programmer needs to think about which parts of their computation can be autotuned. For
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
is launched, the following code is added at the beginning of kernel source code: #define unroll_factor 2. The definitions can be used
to alter kernel functionality based on tuning parameter values.

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
OpenCL). The indices for platforms and devices are assigned by KTT - they can be retrieved through PlatformInfo and DeviceInfo structures. These
structures also contain some other useful information such as list of supported extensions, global memory size, number of available compute units and
more. Note that the assigned indices remain the same when autotuning applications are launched multiple times on the same computer. They change only
when the hardware configuration is changed (e.g., new GPU is added, old GPU is removed, device driver is reinstalled). Also note, that the indices
may not be the same across multiple compute APIs (e.g., index for certain device may differ under OpenCL and CUDA).

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

Todo

#### Subsection

Todo

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
