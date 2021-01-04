#ifdef KTT_PLATFORM_CUDA

#include <stdexcept>
#include <compute_engine/cuda/cuda_context.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

CUDAContext::CUDAContext(const CUdevice device) :
    device(device),
    owningContext(true)
{
    checkCUDAError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device), "cuCtxCreate");
}

CUDAContext::CUDAContext(UserContext context) :
    owningContext(false)
{
    this->context = static_cast<CUcontext>(context);

    if (this->context == nullptr)
    {
        throw std::runtime_error("The provided user CUDA context is not valid");
    }

    checkCUDAError(cuCtxGetDevice(&device), "cuCtxGetDevice");
}

CUDAContext::~CUDAContext()
{
    if (owningContext)
    {
        checkCUDAError(cuCtxDestroy(context), "cuCtxDestroy");
    }
}

CUcontext CUDAContext::getContext() const
{
    return context;
}

CUdevice CUDAContext::getDevice() const
{
    return device;
}

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
