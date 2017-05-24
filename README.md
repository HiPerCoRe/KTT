KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ kernel tuning library based on [CLTune project](https://github.com/CNugteren/CLTune).
It has API similar to that of CLTune, however, unlike CLTune, it also provides greater degree of customization and control over kernel tuning process.
This includes ability to write tuning manipulators, which are classes that can be utilized to launch custom C++ code before or after individual kernel executions.
This can be used, for example, to run some part of a computation on CPU or launch kernels iteratively.

Project is currently in beta stage with all of the baseline functionality available.

Prerequisites  
-------------

The prerequisites to build KTT are:

* C++14 compiler, for example:
    - Clang 3.4 or newer
    - GCC 5.0 or newer
    - MSVC 19.0 (Visual Studio 2015) or newer
* OpenCL library, supported SDKs are:
    - AMD APP SDK
    - Intel SDK for OpenCL
    - NVIDIA CUDA Toolkit
* [Premake 5](https://premake.github.io/download.html) (alpha 11 or newer)

Building KTT
------------

KTT can be built as a static library using command line build tool premake5 (running from
inside KTT root folder). Currently supported operating systems are Linux and Windows.

* Build under Linux:
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config=release` to build the project
    
* Build under Windows:
    - run `premake5.exe vs2015` (or `vs2017`) to generate Visual Studio project files
    - open generated .sln file and build the project inside Visual Studio

Documentation
-------------

Documentation for KTT API can be found [here](https://github.com/Fillo7/KTT/blob/master/documentation/ktt_api.md).

Examples
--------

Examples showcasing KTT functionality are located inside examples folder.
List of currently available examples:

* `opencl_info`: basic example showing how to retrieve detailed information about OpenCL platforms and devices through KTT API
* `simple`: basic example showcasing how to setup kernel run with KTT framework, no actual autotuning is done
* `coulomb_sum`: advanced example which uses several parameters, thread modifiers and constraints

Tests
-----

Basic unit tests are located inside tests folder and are built together with the library.
These can be run to ensure that library methods work correctly on the current platform.
