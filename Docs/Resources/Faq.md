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
A: List of compatible compilers can be found in readme on [main KTT Github page](https://github.com/HiPerCoRe/KTT).
If you are unable to build KTT with compatible compiler (generally any compiler which supports C++17),
you can report a bug [here](https://github.com/HiPerCoRe/KTT/issues).

Using KTT
---------

**Q: I've ported my native OpenCL / CUDA application to KTT but it crashes at runtime.**<br/>
A: KTT checks for correct usage of the API during runtime. If it detects a problem, an exception
is thrown. In most cases, the exception also contains an error message which describes the problem.
If the exception message is not helpful and you believe that you are using the API correctly, you
can report a bug [here](https://github.com/HiPerCoRe/KTT/issues).

**Q: I have an application which performs some part of the computation directly in C/C++ code and
utilizes iterative kernel launches. Can such application be ported to KTT?**<br/>
A: Yes, in this case you need to utilize kernel launcher and compute interface API, which is fully
documented. You can also read some of the examples which already utilize kernel launchers (e.g.,
[Reduction](https://github.com/HiPerCoRe/KTT/tree/master/Examples/Reduction),
[CoulombSum3dIterative](https://github.com/HiPerCoRe/KTT/tree/master/Examples/CoulombSum3dIterative)).

**Q: Running my application with KTT uses much more memory than running it natively.**<br/>
A: KTT by default makes a copy of all buffers that are added to tuner with AddArgumentVector() method.
This makes it safe for user to modify the original buffer without affecting the tuner. However, it
also increases the memory usage. If you are fine with tuner accessing your buffer directly, you can use
overloaded version of AddArgumentVector() method, which allows you to customize handling of buffers
by tuner. You may also want to read KTT buffer types diagram located [here](https://github.com/HiPerCoRe/KTT/tree/master/Docs/Resources).
in order to find out differences between various buffer configuration options.
