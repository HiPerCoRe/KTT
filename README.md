KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ tuning library for OpenCL and CUDA kernels. It is based on [CLTune project](https://github.com/CNugteren/CLTune).
It has API similar to that of CLTune, however, unlike CLTune, it also provides greater degree of customization and control over kernel tuning process.
This includes ability to write tuning manipulators, which are classes that can be utilized to launch custom C++ code before or after individual kernel runs.
This can be used, for example, to run some part of a computation on CPU or launch kernels iteratively.

Project is currently in beta stage with all of the baseline functionality available.

Prerequisites  
-------------

The prerequisites to build KTT are:

* C++14 compiler, for example:
    - Clang 3.4 or newer
    - GCC 5.0 or newer
    - MSVC 19.0 (Visual Studio 2015) or newer
* OpenCL or CUDA library, supported SDKs are:
    - AMD APP SDK
    - Intel SDK for OpenCL
    - NVIDIA CUDA Toolkit
* [Premake 5](https://premake.github.io/download.html) (alpha 11 or newer)

Building KTT
------------

KTT can be built as a dynamic (shared) library using command line build tool Premake.
Currently supported operating systems are Linux and Windows.

* Build under Linux (inside KTT root folder):
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config=release` to build the project
    
* Build under Windows (inside KTT root folder):
    - run `premake5.exe vs2015` (or `premake5.exe vs2017`) to generate Visual Studio project files
    - open generated .sln file and build the project inside Visual Studio

It is possible to specify custom build directory, eg. `premake5 --outdir=my_dir gmake`.
Default build directory is `build`.

When multiple SDKs are installed on a system, it is possible to specify which SDK should be used for building
the KTT library, eg. `premake5 --platform=amd gmake`.

If current platform is Nvidia, support for CUDA functionality will be automatically included as well.
It is possible to disable CUDA compilation, eg. `premake5 --no-cuda gmake`.
    
Documentation
-------------

Documentation for KTT API can be found [here](https://github.com/Fillo7/KTT/blob/master/documentation/ktt_api.md).

Examples
--------

Examples showcasing KTT functionality are located inside examples folder.
List of currently available examples:

* `opencl_info`: basic example showing how to retrieve detailed information about OpenCL platforms and devices through KTT API
* `simple`: basic example showcasing how to run simple kernel with KTT framework, utilizes reference class, no actual autotuning is done
* `coulomb_sum`: advanced example which utilizes large number of tuning parameters, thread modifiers and constraints
* `coulomb_sum_3d`: 3D version of previous example
* `reduction`: advanced example which utilizes reference class, tuning manipulator and several tuning parameters
* `simple_cuda`: version of simple example which utilizes CUDA compute API

It is possible to disable compilation of examples, eg. `premake5 --no-examples gmake`.

Tests
-----

Basic unit tests are located inside tests folder and are built together with the library.
These can be run to ensure that library methods work correctly on the current platform.
In order to enable unit tests compilation, corresponding Premake argument needs to be used, eg. `premake5 --tests gmake`.
