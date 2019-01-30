#pragma once

#include <string>
#include <cuda.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUDAKernel
{
public:
    explicit CUDAKernel(const std::string& ptxSource, const std::string& kernelName) :
        kernelName(kernelName)
    {
        checkCUDAError(cuModuleLoadDataEx(&module, &ptxSource[0], 0, nullptr, nullptr), "cuModuleLoadDataEx");
        checkCUDAError(cuModuleGetFunction(&kernel, module, &kernelName[0]), "cuModuleGetFunction");
    }

    ~CUDAKernel()
    {
        checkCUDAError(cuModuleUnload(module), "cuModuleUnload");
    }

    CUmodule getModule() const
    {
        return module;
    }

    const std::string& getKernelName() const
    {
        return kernelName;
    }

    CUfunction getKernel() const
    {
        return kernel;
    }

private:
    CUmodule module;
    std::string kernelName;
    CUfunction kernel;
};

} // namespace ktt
