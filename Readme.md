KTT - Kernel Tuning Toolkit
===========================
<img src="https://github.com/HiPerCoRe/KTT/blob/master/Docs/Resources/KttLogo.png" width="425" height="150"/>

KTT is an auto-tuning framework for OpenCL, CUDA kernels and GLSL compute shaders. Version 2.0 which contains major
API overhaul as well as new features and improvements is now available.

Main features
-------------
* Ability to define kernel tuning parameters such as kernel thread sizes, vector data types and loop unroll factors
in order to optimize computation for a particular device.
* Support for iterative kernel launches and composite kernels.
* Support for multiple compute queues and asynchronous operations.
* Support for online auto-tuning - kernel tuning combined with regular kernel running.
* Ability to automatically ensure correctness of tuned computation with reference kernel or C++ function.
* Support for multiple compute APIs, switching between CUDA, OpenCL and Vulkan requires only minor changes in C++ code
(e.g., changing the kernel source file), no library recompilation is needed.
* Large number of customization options, including support for kernel arguments with user-defined data types,
ability to change kernel compiler flags and more.

Getting started
---------------
* Documentation for KTT API can be found [here](https://hipercore.github.io/KTT/).
* KTT FAQ can be found [here](https://hipercore.github.io/KTT/md__docs__resources__faq.html).
* The newest release of KTT framework can be found [here](https://github.com/HiPerCoRe/KTT/releases).
* Prebuilt binaries are not provided due to many different combinations of compute APIs and build options available.
Please check the `Building KTT` section for detailed instructions on how to perform a build.

Tutorials
---------
Tutorials are short examples which serve as an introduction to KTT framework. Each tutorial covers a specific part of
the API. All tutorials are available for both OpenCL and CUDA backends. Most of the tutorials are also available for
Vulkan. Tutorials assume that reader has some knowledge about C++ and GPU programming. List of the currently available
tutorials:

* `Info`: Retrieving information about compute API platforms and devices through KTT API.
* `KernelRunning`: Running simple kernel with KTT framework and retrieving output.
* `KernelTuning`: Simple kernel tuning using small number of tuning parameters and reference computation to validate output.
* `CustomArgumentTypes`: Usage of kernel arguments with custom data types and validating the output with value comparator.
* `ComputeApiInitializer`: Providing tuner with custom compute context, queues and buffers.
* `VectorArgumentCustomization`: Showcasing different usage options for vector kernel arguments.

Examples
--------
Examples showcase how KTT framework could be utilized in real-world scenarios. They are more complex than tutorials and
assume that reader is familiar with KTT API. List of some of the currently available examples:

* `CoulombSum2d`: Tuning of electrostatic potential map computation, focuses on a single slice.
* `CoulombSum3dIterative`: 3D version of previous example, utilizes kernel from 2D version and launches it iteratively.
* `CoulombSum3d`: Alternative to iterative version, utilizes kernel which computes the entire map in single invocation.
* `Nbody`: Tuning of N-body simulation.
* `Reduction`: Tuning of vector reduction, launches a kernel iteratively.
* `Sort`: Radix sort example, combines multiple kernels into composite kernel.
* `Bicg`: Biconjugate gradients method example, features reference computation, composite kernels and constraints.

Building KTT
------------
* KTT can be built as a dynamic (shared) library using command line build tool Premake. Currently supported operating
systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++17 compiler, for example Clang 7.0, GCC 9.1, MSVC 14.16 (Visual Studio 2017) or newer
    - OpenCL, CUDA or Vulkan library, supported SDKs are AMD OCL SDK, Intel SDK for OpenCL, NVIDIA CUDA Toolkit
      and Vulkan SDK
    - [Premake 5](https://premake.github.io/download)
    
* Build under Linux (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables
    - run `./premake5 gmake` to generate makefile
    - run `cd Build` to get inside build directory
    - afterwards run `make config={configuration}_{architecture}` to build the project (e.g., `make config=release_x86_64`)
    
* Build under Windows (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables, this should be done automatically
    during SDK installation
    - run `premake5.exe vs20xx` (e.g., `premake5.exe vs2019`) to generate Visual Studio project files
    - open generated solution file and build the project inside Visual Studio

* The following build options are available:
    - `--outdir=path` specifies custom build directory, default build directory is `Build`
    - `--platform=vendor` specifies SDK used for building KTT, useful when multiple SDKs are installed
    - `--profiling=library` enables compilation of kernel profiling functionality using specified library
    - `--vulkan` enables compilation of experimental Vulkan backend
    - `--no-examples` disables compilation of examples
    - `--no-tutorials` disables compilation of tutorials
    - `--tests` enables compilation of unit tests
    - `--no-cuda` disables inclusion of CUDA API during compilation, only affects Nvidia platform
    - `--no-opencl` disables inclusion of OpenCL API during compilation

* KTT and applications utilizing it rely on external dynamic (shared) libraries in order to work correctly. There are
  multiple ways to provide access to these libraries, e.g., copying given library inside application folder or adding the
  containing folder to library path (example for Linux: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/shared/library).
  Libraries which are bundled with device drivers are usually visible by default. The list of libraries currently utilized
  by KTT:
    - `OpenCL` distributed with specific device drivers (OpenCL only)
    - `cuda` distributed with specific device drivers (CUDA only)
    - `nvrtc` distributed with specific device drivers (CUDA only)
    - `cupti` bundled with Nvidia CUDA Toolkit (CUDA profiling only)
    - `nvperf_host` bundled with Nvidia CUDA Toolkit (new CUDA profiling only)
    - `nvperf_target` bundled with Nvidia CUDA Toolkit (new CUDA profiling only)
    - `GPUPerfAPICL` bundled with KTT distribution (AMD OpenCL profiling only)
    - `vulkan` distributed with specific device drivers (Vulkan only)
    - `shaderc_shared` bundled with Vulkan SDK (Vulkan only)
    
Related projects
----------------
KTT API is based on [CLTune project](https://github.com/CNugteren/CLTune). Certain parts of the API are similar to CLTune,
however internal structure is completely rewritten from scratch. The ClTuneGemm and ClTuneConvolution examples are adopted from CLTune.

KTT search space generation and tuning configuration storage techniques are derived from [ATF project](https://dl.acm.org/doi/10.1145/3427093).
Certain modifications were made to the original ATF algorithms due to differences in API and available framework features. The examples stored in AtfSamples folder are adopted from ATF.

How to cite
-----------
F. Petrovič et al. [A benchmark set of highly-efficient CUDA and OpenCL kernels and its dynamic autotuning with Kernel Tuning Toolkit](https://www.sciencedirect.com/science/article/abs/pii/S0167739X19327360). In Future Generation Computer Systems, Volume 108, 2020.
