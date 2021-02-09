KTT - Kernel Tuning Toolkit
===========================
<img src="https://github.com/Fillo7/KTT/blob/master/docs/resources/ktt_logo.png" width="425" height="150"/>

KTT is a tuning framework for OpenCL, CUDA kernels and GLSL compute shaders. Version 1.3 which introduces public searcher
API and user-provided compute queues and buffers is now available.

Main features
-------------
* Ability to define kernel tuning parameters such as kernel thread sizes, vector data types and loop unroll factors
in order to optimize computation for a particular device
* Support for iterative kernel launches and composite kernels
* Support for multiple compute queues and asynchronous operations
* Support for online auto-tuning - kernel tuning combined with regular kernel running
* Ability to automatically ensure correctness of tuned computation with reference kernel or C++ function
* Support for multiple compute APIs, switching between CUDA, OpenCL and Vulkan requires only minor changes in C++ code
(e.g. changing the kernel source file), no library recompilation is needed
* Large number of customization options, including support for kernel arguments with user-defined data types,
ability to change kernel compiler flags and more

Getting started
---------------
* Documentation for KTT API can be found [here](https://hipercore.github.io/KTT/).
* The newest release of KTT framework can be found [here](https://github.com/HiPerCoRe/KTT/releases).
* Prebuilt binaries are not provided due to many different combinations of compute APIs and build options available.
Please check the `Building KTT` section for detailed instructions on how to perform a build.

Tutorials
---------
Tutorials are short examples which serve as an introduction to KTT framework. Each tutorial covers a specific part of
the API. All tutorials are available for both OpenCL and CUDA backends. Most of the tutorials are also available for
Vulkan. Tutorials assume that reader has some knowledge about C++ and GPU programming. List of the currently available
tutorials:

* `compute_api_info`: Retrieving information about compute API platforms and devices through KTT API.
* `running_kernel`: Running simple kernel with KTT framework and retrieving output.
* `tuning_kernel_simple`: Simple kernel tuning using small number of tuning parameters and reference class to ensure
correctness of computation.
* `custom_kernel_arguments`: Usage of kernel arguments with custom data types and validating the output with argument
comparator.
* `user_tuner_initializer`: Providing tuner with custom compute context, queues and buffers.

Examples
--------
Examples showcase how KTT framework could be utilized in real-world scenarios. They are more complex than tutorials and
assume that reader is familiar with KTT API. List of some of the currently available examples:

* `coulomb_sum_2d`: Tuning of electrostatic potential map computation, focuses on a single slice.
* `coulomb_sum_3d_iterative`: 3D version of previous example, utilizes kernel from 2D version and launches it iteratively.
* `coulomb_sum_3d`: Alternative to iterative version, utilizes kernel which computes entire map in single invocation.
* `nbody`: Tuning of N-body simulation.
* `reduction`: Tuning of vector reduction, launches a kernel iteratively.
* `sort`: Radix sort example, combines multiple kernels into kernel composition.
* `bicg`: Biconjugate gradients method example, features reference class, kernel compositions and constraints.

Building KTT
------------
* KTT can be built as a dynamic (shared) library using command line build tool Premake. Currently supported operating
systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++14 compiler, for example Clang 3.5, GCC 5.0, MSVC 19.0 (Visual Studio 2015) or newer
    - OpenCL, CUDA or Vulkan library, supported SDKs are AMD APP SDK, Intel SDK for OpenCL, NVIDIA CUDA Toolkit
      and Vulkan SDK
    - [Premake 5](https://premake.github.io/download.html)
    
* Build under Linux (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config={configuration}_{architecture}` to build the project (e.g. `make config=release_x86_64`)
    
* Build under Windows (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables, this should be done automatically
    during SDK installation
    - run `premake5.exe vs20xx` (e.g. `premake5.exe vs2019`) to generate Visual Studio project files
    - open generated solution file and build the project inside Visual Studio

* The following build options are available:
    - `--outdir=path` specifies custom build directory, default build directory is `build`
    - `--platform=vendor` specifies SDK used for building KTT, useful when multiple SDKs are installed
    - `--profiling=library` enables compilation of kernel profiling functionality using specified library
    - `--vulkan` enables compilation of experimental Vulkan backend
    - `--no-examples` disables compilation of examples
    - `--no-tutorials` disables compilation of tutorials
    - `--tests` enables compilation of unit tests
    - `--no-cuda` disables inclusion of CUDA API during compilation, only affects Nvidia platform
    - `--no-opencl` disables inclusion of OpenCL API during compilation

* KTT and applications utilizing it rely on external dynamic (shared) libraries in order to work correctly. There are
  multiple ways to provide access to these libraries, e.g. copying given library inside application folder or adding the
  containing folder to library path (example for Linux: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/shared/library).
  Libraries which are bundled with device drivers are usually visible by default. List of the libraries currently utilized
  by KTT:
    - `OpenCL` distributed with specific device drivers (OpenCL only)
    - `cuda` distributed with specific device drivers (CUDA only)
    - `nvrtc` distributed with specific device drivers (CUDA only)
    - `cupti` bundled with Nvidia CUDA Toolkit (CUDA profiling only)
    - `vulkan` distributed with specific device drivers (Vulkan only)
    - `shaderc_shared` bundled with KTT distribution (Vulkan only)
    
Original project
----------------
KTT is based on [CLTune project](https://github.com/CNugteren/CLTune). Some parts of KTT API are similar to CLTune API,
however internal structure was almost completely rewritten from scratch. Portions of code for following features were ported
from CLTune:
* Annealing searcher
* Generating of kernel configurations
* Tuning parameter constraints
