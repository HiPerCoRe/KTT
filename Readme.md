KTT - Kernel Tuning Toolkit
===========================
<img src="https://github.com/HiPerCoRe/KTT/blob/master/Docs/Resources/KttLogo.png" width="425" height="150"/>

KTT is an autotuning framework for OpenCL, CUDA kernels and GLSL compute shaders. Version 2.1 which introduces
API bindings for Python and new onboarding guide is now available.

Main features
-------------
* Ability to define kernel tuning parameters such as kernel thread sizes, vector data types and loop unroll factors
to optimize computation for a particular device.
* Support for iterative kernel launches and composite kernels.
* Support for multiple compute queues and asynchronous operations.
* Support for online auto-tuning - kernel tuning combined with regular kernel running.
* Ability to automatically ensure the correctness of tuned computation with reference kernel or C++ function.
* Support for multiple compute APIs, switching between CUDA, OpenCL and Vulkan requires only minor changes in C++ code
(e.g., changing the kernel source file), no library recompilation is needed.
* Public API available in C++ (native) and Python (bindings).
* Many customization options, including support for kernel arguments with user-defined data types, ability to change
kernel compiler flags and more.

Getting started
---------------
* Introductory guide to KTT can be found [here](https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md).
* Full documentation for KTT API can be found [here](https://hipercore.github.io/KTT/).
* KTT FAQ can be found [here](https://hipercore.github.io/KTT/md__docs__resources__faq.html).
* The newest release of the KTT framework can be found [here](https://github.com/HiPerCoRe/KTT/releases).
* Prebuilt binaries are not provided due to many different combinations of compute APIs and build options available.
The `Building KTT` section contains detailed instructions on how to perform a build.

Tutorials
---------
Tutorials are short examples that serve as an introduction to the KTT framework. Each tutorial covers a specific part of
the API. All tutorials are available for both OpenCL and CUDA backends. Most of the tutorials are also available for
Vulkan. Tutorials assume that the reader has some knowledge about C++ and GPU programming. List of the currently available
tutorials:

* `Info`: Retrieving information about compute API platforms and devices through KTT API.
* `KernelRunning`: Running simple kernel with KTT framework and retrieving output.
* `KernelTuning`: Simple kernel tuning using a small number of tuning parameters and reference computation to validate output.
* `CustomArgumentTypes`: Usage of kernel arguments with custom data types and validating the output with value comparator.
* `ComputeApiInitializer`: Providing tuner with custom compute context, queues and buffers.
* `VectorArgumentCustomization`: Showcasing different usage options for vector kernel arguments.
* `PythonInterfaces`: Implementing custom searchers and stop conditions in Python, which can afterward be used with the tuner.

Examples
--------
Examples showcase how the KTT framework could be utilized in real-world scenarios. They are more complex than tutorials and
assume that the reader is familiar with KTT API. List of some of the currently available examples:

* `CoulombSum2d`: Tuning of electrostatic potential map computation, focuses on a single slice.
* `CoulombSum3dIterative`: 3D version of the previous example, utilizes kernel from 2D version and launches it iteratively.
* `CoulombSum3d`: Alternative to iterative version, utilizes kernel which computes the entire map in a single invocation.
* `Nbody`: Tuning of N-body simulation.
* `Reduction`: Tuning of vector reduction, launches a kernel iteratively.
* `Sort`: Radix sort example, combines multiple kernels into a composite kernel.
* `Bicg`: Biconjugate gradients method example, features reference computation, composite kernels and constraints.

Building KTT
------------
* KTT can be built as a dynamic (shared) library using the command line build tool Premake. Currently supported operating
systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++17 compiler, for example Clang 7.0, GCC 9.1, MSVC 14.16 (Visual Studio 2017) or newer
    - OpenCL, CUDA or Vulkan library, supported SDKs are AMD OCL SDK, Intel SDK for OpenCL, NVIDIA CUDA Toolkit
      and Vulkan SDK
    - Command line build tool [Premake 5](https://premake.github.io/download)
    - (Optional) Python 3 with NumPy for Python bindings support
    
* Build under Linux (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables
    - run `./premake5 gmake` to generate makefile
    - run `cd Build` to get inside the build directory
    - afterwards run `make config={configuration}_{architecture}` to build the project (e.g., `make config=release_x86_64`)
    
* Build under Windows (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables; this should be done automatically
    during SDK installation
    - run `premake5.exe vs20xx` (e.g., `premake5.exe vs2019`) to generate Visual Studio project files
    - open generated solution file and build the project inside Visual Studio

* The following build options are available:
    - `--outdir=path` Specifies custom build directory. The default build directory is `Build`.
    - `--platform=vendor` Specifies SDK used for building KTT. May be useful when multiple SDKs are installed.
    - `--profiling=library` Enables compilation of kernel profiling functionality using the specified library.
    - `--power-usage` Enables compilation of device power usage collection functionality. This feature is currently supported only on Nvidia platform.
    - `--vulkan` Enables compilation of experimental Vulkan backend.
    - `--python` Enables compilation of Python bindings.
    - `--no-examples` Disables compilation of examples.
    - `--no-tutorials` Disables compilation of tutorials.
    - `--tests` Enables compilation of unit tests.
    - `--no-cuda` Disables the inclusion of CUDA API during compilation. Only affects Nvidia platform.
    - `--no-opencl` Disables the inclusion of OpenCL API during compilation.

* KTT and applications that utilize it rely on external dynamic (shared) libraries to work correctly. There are
  multiple ways to provide access to these libraries, e.g., copying a given library inside the application folder or adding the
  containing folder to the library path (example for Linux: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/shared/library).
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
    
Python bindings
---------------
To be able to use KTT Python API, the KTT module must be built with `--python` option. For the build option to work, access to Python
development headers and library must be provided under environment variables `PYTHON_HEADERS` and `PYTHON_LIB` respectively. Once the
build is finished, in addition to the regular C++ module, a Python module will be created (named `pyktt.pyd` under Windows, `pyktt.so`
under Linux). This module can be imported into Python programs in the same way as regular modules. Note that Python must have access to
all modules which depend on the KTT module (e.g., various profiling libraries), otherwise the loading will fail.

Related projects
----------------
KTT API is based on [CLTune project](https://github.com/CNugteren/CLTune). Certain parts of the API are similar to CLTune. However, the internal
structure is completely rewritten from scratch. The ClTuneGemm and ClTuneConvolution examples are adopted from CLTune.

KTT search space generation and tuning configuration storage techniques are derived from [ATF project](https://dl.acm.org/doi/10.1145/3427093).
Due to differences in API and available framework features, certain modifications were made to the original ATF algorithms. The examples stored
in AtfSamples folder are adopted from ATF.

How to cite
-----------
F. Petrovič et al. [A benchmark set of highly-efficient CUDA and OpenCL kernels and its dynamic autotuning with Kernel Tuning Toolkit](https://www.sciencedirect.com/science/article/abs/pii/S0167739X19327360). In Future Generation Computer Systems, Volume 108, 2020.
