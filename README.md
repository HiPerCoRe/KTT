KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ tuning library for OpenCL and CUDA kernels.
Project is currently in late beta stage with all of the baseline functionality available.

Main features
-------------
* Ability to define kernel tuning parameters like thread sizes, vector data types and loop unroll factors in order to fit computation
for particular device
* High-level API which is easier to use than low-level APIs like OpenCL or CUDA driver API
* Shared C++ code betweem multiple compute APIs, switching between CUDA and OpenCL involves only minor changes to C++ code
(eg. changing path to kernel file)
* Ability to optionally gain greater control over computation by using tuning manipulator, which allows, for example to utilize iterative
kernel launches or run some part of a computation directly in C++ code
* Ability to automatically ensure correctness of tuned computation by providing reference kernel or C++ method
* Large number of utility methods and customization options, for example ability to specify custom tolerance threshold for floating-point
argument validation, ability to specify output target for printing results (C++ stream or file), ability to easily retrieve information about
available devices, etc.
* No direct usage of vendor specific SDKs is needed, only corresponding device drivers have to be installed

Original project
----------------

KTT is based on [CLTune project](https://github.com/CNugteren/CLTune). Basic KTT API functionality is similar to that of CLTune,
however internal structure was almost completely rewritten from scratch. Parts of code for following features were ported from CLTune:
* Searchers (mainly PSO and annealing searcher)
* Generating of kernel configurations
* Tuning parameter constraints

Getting started
---------------

* Newest version of KTT library can be found [here](https://github.com/Fillo7/KTT/releases).
* Prebuilt binaries are currently available only for some platforms. Other platforms require manual build.
* Documentation for KTT API can be found [here](https://github.com/Fillo7/KTT/blob/master/documentation/ktt_api.md).

Examples
--------

Examples showcasing KTT functionality are located inside examples folder.
List of currently available examples:

* `opencl_info`: basic example showing how to retrieve detailed information about OpenCL platforms and devices through KTT API
* `simple`: basic example showing how to run simple kernel with KTT framework, utilizes reference class, no actual autotuning is done
* `coulomb_sum_2d`: advanced example which utilizes large number of tuning parameters, thread modifiers and constraints
* `coulomb_sum_3d_iterative`: 3D version of previous example, utilizes tuning manipulator to iteratively launch 2D kernel
* `coulomb_sum_3d`: alternative to iterative version, utilizes several tuning parameters and reference kernel
* `nbody`: advanced example which utilizes tuning parameters, multiple constraints and validation of multiple arguments with reference kernel
* `reduction`: advanced example which utilizes reference class, tuning manipulator and several tuning parameters

It is possible to disable compilation of examples, eg. `premake5 --no-examples gmake`.

Building KTT
------------

* KTT can be built as a dynamic (shared) library using command line build tool Premake.
Currently supported operating systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++14 compiler, for example Clang 3.4, GCC 5.0, MSVC 19.0 (Visual Studio 2015) or newer
    - OpenCL or CUDA library, supported SDKs are AMD APP SDK, Intel SDK for OpenCL and NVIDIA CUDA Toolkit
    - [Premake 5](https://premake.github.io/download.html) (alpha 11 or newer)

* Build under Linux (inside KTT root folder):
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config=release` to build the project
    
* Build under Windows (inside KTT root folder):
    - run `premake5.exe vs2015` (or `premake5.exe vs2017`) to generate Visual Studio project files
    - open generated .sln file and build the project inside Visual Studio

* It is possible to specify custom build directory, eg. `premake5 --outdir=my_dir gmake`. Default build directory is `build`.

* When multiple SDKs are installed on a system, it is possible to specify which SDK should be used for building the KTT library,
eg. `premake5 --platform=amd gmake`.

* If current platform is Nvidia, support for CUDA functionality will be automatically included as well.
It is possible to disable CUDA compilation, eg. `premake5 --no-cuda gmake`.

* Basic unit tests can be built together with the library. These can be run to ensure that library works correctly on current platform.
In order to enable unit tests compilation, corresponding Premake argument needs to be used, eg. `premake5 --tests gmake`.
