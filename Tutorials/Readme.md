# Tutorials

This folder contains tutorials showcasing various KTT features, sorted by complexity level 
and the likelyhood of being necessary for a typical project.
Each tutorial is a self-contained project. 
Most are written in multiple versions -- C++ host code with a Cuda/OpenCL kernel, etc.

Tutorial projects have comments that delimit which parts of the code are similar to a different tutorial and which are unique.

Within each level, the tutorial folders are numbered from 01 in the reading order used below.

## Basic
Showcases basic features that will likely be involved in nearly every project.
- TunerInitialization: Basic tuner initialization. Also a sanity check that KTT is working correctly and can access the 
    accelerator devices.
- KernelTuning: Basic kernel tuning tutorial; shows kernel arguments, reference computation, 
    tuning parameters, thread modifiers, and offline tuning. Has versions with C++ host code for tuning CUDA, OpenCL, and Vulkan,
    with Python host code for tuning CUDA, and with a JSON script for tuning CUDA.
- ProcessingKernelResults: A few basic ways that one can work with the kernel results after tuning.
- StopConditions: Showcases the use of stop conditions on a very simple kernel. Includes a union stop condition.
- TuningConstraints: Showcases the use of tuning constraints to filter out configurations that would fail or be suboptimal,
    demonstrated on a matrix transpose kernel with work-group size constrained through two tuning parameters.
- MultipleBackends: Program adapted to work with multiple different compute APIs, showing how KTT is designed in a way that lets
    these variants share most of the code.
- SearcherChoice: KTT allows one to choose from several pre-defined searchers.
- AlternateLanguages: Shows that KTT can also be used from Python or JSON scripts.

<!-- perhaps a tutorial specifically about reference kernel/computation -->

## Intermediate
Showcases features that are good to know about, but might not be necessary for all projects.
- KernelRunning: Shows that KTT can also run kernels <!-- TODO: Merge with dynamic tuning? Perhaps dedicate to transferring data 
                                                                between kernel and host -->
- DynamicTuning: An application can tune during runtime with KTT's dynamic tuning. It is possible to split processing time
    between exploration of different configurations and exploitation of the best known one. One can also use the tuning runs
    to already process actual input. Lastly, tuning can be restarted if input shape changes significantly enough, though that is
    outside the scope of this tutorial.
- KernelLauncher: KTT allows the user to define a custom kernel launcher. This tutorial shows how one can split the work between
    GPU and CPU with this.
- CompositeKernel: A kernel can also be made of several kernel definitions that are tuned together. This tutorial splits a blur
    into two passes and runs them in order with a kernel launcher. The work-group size of every definition is tuned on its own, and
    a common tuning parameter chooses in which direction the intermediate result is stored. There is also a practical explanation on
    the difference between kernel and kernel definition.
- CompilerTuning: Explains the difference between compiler tuning and separate compiler tuning on a sample C++ kernel.

<!-- TODO: compiler options -- those are already in MultipleBackends, should i move to basic? does it even need its own tutorial?;
           database stuff;
           profiling searcher (or move to advanced?) -->

## Advanced
Showcases features that are unlikely to be necessary, but possibly useful for more advanced projects. 
Mostly customization of things usually hidden in KTT's internals.
- CustomArgumentTypes: Shows that kernel arguments can also be of user-defined data types (e.g., structs);
    a user-defined comparison function is used by the tuner to validate results. Has versions with C++ host code for tuning CUDA and OpenCL.
- ComputeApiInitializer: Shows that compute API objects (context, streams, buffers) can be created externally and imported
    into KTT through a ComputeApiInitializer; existing device buffers can be added as arguments by handle.
    Has versions with C++ host code for tuning CUDA and OpenCL.
- VectorArgumentCustomization: Shows customization of vector argument handling -- argument memory location (device/host),
    buffer management type (framework/user, with manual uploading and clearing in a kernel launcher), and referencing
    user-provided buffers directly instead of copying them. Has versions with C++ host code for tuning CUDA and OpenCL.
- PythonInterfaces: Shows that it is possible to pass Python functions
    to KTT, e.g. to define a custom searcher or stop condition.

<!-- TODO: custom searcher, stop condition, simulated tuning(?) -- when writing your own searcher you can easily see how fast it converges,
    precise measurement (advanced?), power consumption optimization (advanced?), etc -->