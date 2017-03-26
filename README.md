KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ kernel tuning library based on [CLTune project](https://github.com/CNugteren/CLTune).
It aims to provide API similar to that of CLTune, however, unlike CLTune, it also aims to provide
greater degree of customization and control over kernel tuning process. This includes ability to
write custom classes, which can launch custom C++ code before or after individual kernel executions
in order to run some part of computation on a CPU.

Project is currently under development.

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

KTT can be built as a static library using premake. Currently supported operating systems
are Linux and Windows.

* Build under Linux (using gmake, inside KTT root folder):
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config=release`
    
* Build under Windows (using Visual Studio 2015, inside KTT root folder):
    - run `premake5.exe vs2015` to generate Visual Studio project file
    - open generated file and build project inside Visual Studio

Examples
--------

Examples showcasing KTT functionality are located inside examples folder. List of currently
available examples:

* `opencl_info` - example showing how to retrieve detailed information about OpenCL platforms
  and devices through KTT API
* `simple` - example showcasing how to setup kernel run with KTT framework, no autotuning is done
