KTT FAQ
=======

This file provides answers to common questions that users might have involving the usage of KTT framework.

Building KTT
------------

**Q: During project file generation, Premake prints error that compute API libraries were not found.**<br/>
A: There are two likely reasons for this. First, you need to make sure that compute SDK provided by
your device vendor (eg. CUDA Toolkit) is installed correctly on your machine. Second, you need to set
path to the SDK in your environment variables (eg. setting path to CUDA Toolkit on Linux:
`export CUDA_PATH=/path/to/cuda/toolkit`). On Windows, the path to SDK is usually set automatically
during SDK installation.

**Q: I'm getting compilation errors during KTT build.**<br/>
A: List of compatible compilers can be found in readme on [main KTT Github page](https://github.com/Fillo7/KTT).
If you are unable to build KTT with compatible compiler (generally any compiler which supports C++14),
you can report a bug [here](https://github.com/Fillo7/KTT/issues).

Using KTT
---------

**Q: I've ported my native OpenCL / CUDA application to KTT but it crashes at runtime.**<br/>
A: KTT checks for correct usage of the API during runtime. If it detects a problem, an exception
is thrown. In most cases, the exception also contains an error message which describes the problem.
If the exception message is not helpful and you believe that you are using the API correctly, you
can report a bug [here](https://github.com/Fillo7/KTT/issues).

**Q: I have an application which performs some part of the computation directly in C/C++ code and
utilizes iterative kernel launches. Can such application be ported to KTT?**<br/>
A: Yes, in this case you need to utilize tuning manipulator API, which is fully documented. You can
also check out some of the examples which already utilize tuning manipulator (eg.
[reduction](https://github.com/Fillo7/KTT/tree/master/examples/reduction),
[coulomb_sum_3d_iterative](https://github.com/Fillo7/KTT/tree/master/examples/coulomb_sum_3d_iterative)).

**Q: Running my application with KTT uses much more memory than running it natively.**<br/>
A: KTT by default makes a copy of all buffers that are added to tuner with addArgumentVector() method.
This makes it safe for user to modify the original buffer without affecting the tuner. However, it
also doubles the memory usage. If you are fine with tuner accessing your buffer directly, you can use
overloaded version of addArgumentVector() method, which allows you to customize handling of buffers
by tuner. You may also want to read KTT buffer diagram located in documentation folder to find out
differences between various buffer configuration options.

**Q: How do I specify tuning parameters for specific kernels inside a kernel composition?**<br/>
A: You can use methods which affect regular kernels for kernel compositions as well. In case you call
such a method with kernel composition id as an argument, it will simply affect all kernels which are
part of the composition. This works even if the tuning parameter is supposed to affect only a single
kernel inside composition because the parameters which do not affect thread sizes are only added to
kernel source as preprocessor definitions, which can be ignored by specific kernels. Note that for
thread and local memory modifiers, versions of methods which only affect specific kernels inside
compositions are available.
