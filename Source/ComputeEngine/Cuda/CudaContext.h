#pragma once

#ifdef KTT_API_CUDA

#include <cuda.h>

#include <KttTypes.h>

namespace ktt
{

class CudaDevice;

class CudaContext
{
public:
    explicit CudaContext(const CudaDevice& device);
    explicit CudaContext(ComputeContext context);
    ~CudaContext();

    void EnsureThreadContext() const;

    CUcontext GetContext() const;
    CUdevice GetDevice() const;
    bool IsUserOwned() const;

private:
    CUcontext m_Context;
    CUdevice m_Device;
    bool m_OwningContext;
};

} // namespace ktt

#endif // KTT_API_CUDA
