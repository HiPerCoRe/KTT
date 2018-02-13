KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ tuning framework for OpenCL and CUDA kernels. Project is currently in late beta stage with all of the baseline
functionality available.

Main features
-------------
* Ability to define kernel tuning parameters like thread count, vector data types and loop unroll factors in order
to optimize computation for particular device
* Support for iterative kernel launches and composite kernels
* Ability to automatically ensure correctness of tuned computation with reference kernel or C++ function
* Ability to run kernels with low overhead after finding optimal configuration during kernel tuning
* Support for multiple compute APIs, switching between CUDA and OpenCL requires only minor changes in C++ code
(eg. changing the kernel source file), no library recompilation is needed
* Large number of customization options, including ability to specify custom tolerance threshold for floating-point
argument validation, ability to change kernel compiler flags and more

Getting started
---------------

* Documentation for KTT API can be found [here](https://github.com/Fillo7/KTT/blob/master/documentation).
* Newest version of KTT framework can be found [here](https://github.com/Fillo7/KTT/releases).
* Prebuilt binaries are currently available only for some platforms. Other platforms require manual build.
* Prebuilt binaries for Nvidia include both CUDA and OpenCL support, binaries for AMD and Intel include only OpenCL support.

Tutorials
---------

Tutorials are short examples aimed at introducing people to KTT framework. Each tutorial focuses on explaining specific part
of the API. All tutorials are available for both OpenCL and CUDA back-ends. Tutorials assume that reader has some knowledge
about C++ and GPU programming. List of currently available tutorials:

* `compute_api_info`: Tutorial covers retrieving information about compute API platforms and devices through KTT API.
* `running_kernel`: Tutorial covers running simple kernel with KTT framework and retrieving output.
* `tuning_kernel_simple`: Tutorial covers simple kernel tuning using small number of tuning parameters and reference class
to ensure correctness of computation.

Examples
--------

Examples showcase how KTT framework could be utilized in real-world scenarios. Examples are more complex than tutorials and
assume that reader is familiar with KTT API. List of currently available examples:

* `coulomb_sum_2d`: Example which showcases tuning of electrostatic potential map computation, it focuses on a single slice.
* `coulomb_sum_3d_iterative`: 3D version of previous example, utilizes kernel from 2D version and launches it iteratively.
* `coulomb_sum_3d`: Alternative to iterative version, utilizes kernel which computes entire map in single invocation.
* `nbody`: Example which showcases tuning of N-body simulation.
* `reduction`: Example which showcases tuning of vector reduction, launches a kernel iteratively.

Building KTT
------------

* KTT can be built as a dynamic (shared) library using command line build tool Premake. Currently supported operating
systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++14 compiler, for example Clang 3.5, GCC 5.0, MSVC 19.0 (Visual Studio 2015) or newer
    - OpenCL or CUDA library, supported SDKs are AMD APP SDK 3.0, Intel SDK for OpenCL and NVIDIA CUDA Toolkit 7.5 or newer
    - [Premake 5](https://premake.github.io/download.html) (alpha 12 or newer)
    
* Build under Linux (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables
    - run `./premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config={configuration}_{architecture}` to build the project (eg. `make config=release_x86_64`)
    
* Build under Windows (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables, this should be done automatically
    during SDK installation
    - run `premake5.exe vs2015` (or `premake5.exe vs2017`) to generate Visual Studio project files
    - open generated solution file and build the project inside Visual Studio

* Following build options are available:
    - `--outdir=path` specifies custom build directory, default build directory is `build`
    - `--platform=vendor` specifies SDK used for building KTT, useful when multiple SDKs are installed
    - `--no-examples` disables compilation of examples
    - `--no-tutorials` disables compilation of tutorials
    - `--tests` enables compilation of unit tests
    - `--no-cuda` disables inclusion of CUDA API during compilation, only affects Nvidia platform

Original project
----------------

KTT is based on [CLTune project](https://github.com/CNugteren/CLTune). Some parts of KTT API are similar to CLTune API,
however internal structure was almost completely rewritten from scratch. Portions of code for following features were ported
from CLTune:
* PSO and annealing searcher
* Generating of kernel configurations
* Tuning parameter constraints
