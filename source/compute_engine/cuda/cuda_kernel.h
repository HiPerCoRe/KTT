#pragma once

#include <string>
#include "cuda.h"
#include "cuda_utility.h"

namespace ktt
{

class CudaKernel
{
public:
    explicit CudaKernel(const std::string& ptxSource, const std::string& kernelName) :
        kernelName(kernelName)
    {
        checkCudaError(cuModuleLoadDataEx(&module, &ptxSource[0], 0, nullptr, nullptr), "cuModuleLoadDataEx");
        checkCudaError(cuModuleGetFunction(&kernel, module, &kernelName[0]), "cuModuleGetFunction");
    }

    CUmodule getModule() const
    {
        return module;
    }

    std::string getKernelName() const
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
