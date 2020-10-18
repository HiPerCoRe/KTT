#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <cuda.h>
#include <ktt_types.h>

namespace ktt
{

class CUDAContext
{
public:
    explicit CUDAContext(const CUdevice device);
    explicit CUDAContext(UserContext context);
    ~CUDAContext();

    CUcontext getContext() const;
    CUdevice getDevice() const;

private:
    CUcontext context;
    CUdevice device;
    bool owningContext;
};

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
