KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ kernel tuning library based on [CLTune project](https://github.com/CNugteren/CLTune).
It aims to provide API similar to that of CLTune, however, unlike CLTune, it also aims to provide
greater degree of customization and control over kernel tuning process. This includes ability to
write classes, which can be used to launch custom C++ code before or after individual kernel
executions in order to run some part of a computation on CPU.

Project is currently in beta stage with majority of the baseline functionality available.

Prerequisites  
-------------

The prerequisites to build KTT are:

* [premake5](https://premake.github.io/download.html)
* C++14 compiler, for example:
    - Clang 3.4 or newer
    - GCC 5.1 or newer
    - MSVC 19.0 (Visual Studio 2015) or newer
* OpenCL library, supported SDKs are:
    - AMD APP SDK
    - Intel SDK for OpenCL
    - NVIDIA CUDA Toolkit

Building KTT
------------

KTT can be built as a static library using command line build tool premake5 (running from
inside KTT root folder). Currently supported operating systems are Linux and Windows.

* Build under Linux:
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config=release` to build the project
    
* Build under Windows:
    - run `premake5.exe vs2015` to generate Visual Studio project files
    - open generated .sln file and build the project inside Visual Studio

Documentation
-------------

Documentation for KTT API can be found inside documentation folder.

Examples
--------

Examples showcasing KTT functionality are located inside examples folder. List of currently
available examples:

* `opencl_info` - basic example showing how to retrieve detailed information about OpenCL
  platforms and devices through KTT API
* `simple` - basic example showcasing how to setup kernel run with KTT framework, no actual
  autotuning is done
* `coulomb_sum` - advanced example which uses several parameters, thread modifiers
  and constraints

Tests
-----

Basic unit tests are located inside tests folder and are built together with the library.
You can run these to ensure that library methods work correctly on your current platform.
