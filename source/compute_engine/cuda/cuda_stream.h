#pragma once

#include <string>
#include "cuda.h"
#include "cuda_utility.h"
#include "ktt_types.h"

namespace ktt
{

class CudaStream
{
public:
    explicit CudaStream(const QueueId id, const CUcontext context, const CUdevice device) :
        id(id),
        context(context),
        device(device)
    {
        checkCudaError(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
    }

    ~CudaStream()
    {
        checkCudaError(cuStreamDestroy(stream), "cuStreamDestroy");
    }

    QueueId getId() const
    {
        return id;
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
    QueueId id;
    CUcontext context;
    CUdevice device;
    CUstream stream;
};

} // namespace ktt
