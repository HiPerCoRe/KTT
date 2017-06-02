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
        kernelName(kernelName),
        argumentsCount(0)
    {
        checkCudaError(cuModuleLoadDataEx(&module, &ptxSource[0], 0, nullptr, nullptr), std::string("cuModuleLoadDataEx"));
        checkCudaError(cuModuleGetFunction(&kernel, module, &kernelName[0]), std::string("cuModuleGetFunction"));
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

    size_t getArgumentsCount() const
    {
        return argumentsCount;
    }

private:
    CUmodule module;
    std::string kernelName;
    CUfunction kernel;
    size_t argumentsCount;
};

} // namespace ktt
