#pragma once

#include "cuda.h"
#include "cuda_utility.h"

namespace ktt
{

class CudaContext
{
public:
    explicit CudaContext(const CUdevice device) :
        device(device)
    {
        checkCudaError(cuCtxCreate(&context, 0, device), "cuCtxCreate");
    }

    ~CudaContext()
    {
        checkCudaError(cuCtxDestroy(context), "cuCtxDestroy");
    }

    CUdevice getDevice() const
    {
        return device;
    }

    CUcontext getContext() const
    {
        return context;
    }

private:
    CUdevice device;
    CUcontext context;
};

} // namespace ktt
