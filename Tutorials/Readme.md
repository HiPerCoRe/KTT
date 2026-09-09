# Tutorials

This folder contains tutorials showcasing various KTT features, sorted by complexity level 
and the likelyhood of being necessary for a typical project.
Each tutorial is a self-contained project. 
Most are written in multiple versions -- C++ host code with a Cuda/OpenCL kernel, Python host code, etc.

## Basic
Showcases basic features that will likely be involved in nearly every project.
- ComputeApiInfo: Sanity check that KTT is working correctly and can access the accelerator devices.
- KernelTuning: Basic kernel tuning tutorial; shows kernel arguments, reference computation, 
    tuning parameters, thread modifiers, and offline tuning. Has versions with C++ host code for tuning CUDA, OpenCL, and Vulkan,
    with Python host code for tuning CUDA, and with a JSON script for tuning CUDA.  <!-- TODO: split this? -->

<!-- TODO: tuning constraints, stop conditions, cpp tuning(? if there's already python), debugging methods(?) -->

## Intermediate
Showcases features that are good to know about, but might not be necessary for all projects.
- KernelRunning: Shows that KTT can also run kernels <!-- TODO: Merge with dynamic tuning? -->

<!-- TODO: dynamic tuning, groups, kernel launcher, composite kernel, searchers, compiler tuning, compiler options, 
           precise measurement (advanced?), power consumption optimization (advanced?), processing kernel results
-->

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
- ProfileBasedSearcher: Shows a custom Python searcher that guides the search using profiling counters -- it periodically re-runs
    the best configuration with profiling enabled, and uses an ML model to predict profiling counters and score
    the remaining tuning space configurations. (Demonstrated on a real application kernel.)

<!-- TODO: custom searcher, stop condition, simulated tuning(?) etc -->