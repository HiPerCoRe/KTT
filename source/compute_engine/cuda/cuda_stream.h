#pragma once

#include <string>
#include "cuda.h"
#include "cuda_utility.h"

namespace ktt
{

class CudaStream
{
public:
    explicit CudaStream(const CUcontext context, const CUdevice device) :
        context(context),
        device(device)
    {
        checkCudaError(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
    }

    ~CudaStream()
    {
        checkCudaError(cuStreamDestroy(stream), "cuStreamDestroy");
    }

    CUcontext getcontext() const
    {
        return context;
    }

    CUdevice getDevice() const
    {
        return device;
    }

    CUstream getStream() const
    {
        return stream;
    }

private:
    CUcontext context;
    CUdevice device;
    CUstream stream;
};

} // namespace ktt
