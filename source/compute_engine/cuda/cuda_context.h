#pragma once

#include <cuda.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUDAContext
{
public:
    explicit CUDAContext(const CUdevice device) :
        device(device)
    {
        checkCUDAError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device), "cuCtxCreate");
    }

    ~CUDAContext()
    {
        checkCUDAError(cuCtxDestroy(context), "cuCtxDestroy");
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
