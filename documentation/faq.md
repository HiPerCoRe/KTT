KTT FAQ
=======

Building KTT
------------

* Q: During project file generation, Premake prints error that compute API libraries were not found.
* A: There are two likely reasons for this. First, you need to make sure that compute SDK provided by
your device vendor (eg. CUDA Toolkit) is installed correctly on your machine. Second, you need to set
path to the SDK in your environment variables (eg. setting path to CUDA Toolkit on Linux:
'export CUDA_PATH=/path/to/cuda/toolkit'). On Windows, the path to SDK is usually set automatically
during SDK installation.

* Q: I'm getting compilation errors during KTT build.
* A: List of compatible compilers can be found in readme on [main KTT Github page](https://github.com/Fillo7/KTT).
If you are unable to build KTT with compatible compiler (generally any compiler which supports C++14),
you can report a bug [here](https://github.com/Fillo7/KTT/issues).

Using KTT
---------

* Q: I've ported my native OpenCL / CUDA application to KTT but it crashes at runtime.
* A: KTT checks for correct usage of the API during runtime. If it detects a problem, an exception
is thrown. In most cases, the exception also contains an error message which describes the problem.
If the exception message is not helpful and you believe that you are using the API correctly, you
can report a bug [here](https://github.com/Fillo7/KTT/issues).

* Q: I have an application which performs some part of the computation directly in C/C++ code and
utilizes iterative kernel launches. Can such application be ported to KTT?
* A: Yes, in this case you need to utilize tuning manipulator API, which is fully documented. You can
also check out some of the examples which already utilize tuning manipulator (eg.
[reduction](https://github.com/Fillo7/KTT/tree/master/examples/reduction) or
[coulomb_sum_3d_iterative](https://github.com/Fillo7/KTT/tree/master/examples/coulomb_sum_3d_iterative)).
