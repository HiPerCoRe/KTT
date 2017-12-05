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
        checkCudaError(cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC, device), "cuCtxCreate");
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
