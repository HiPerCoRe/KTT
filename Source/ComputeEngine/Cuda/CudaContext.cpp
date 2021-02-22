#ifdef KTT_API_CUDA

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaDevice.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaContext::CudaContext(const CudaDevice& device) :
    m_Device(device.GetDevice()),
    m_OwningContext(true)
{
    Logger::LogDebug("Initializing CUDA context");
    CheckError(cuCtxCreate(&m_Context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, m_Device), "cuCtxCreate");
}

CudaContext::CudaContext(ComputeContext context) :
    m_OwningContext(false)
{
    Logger::LogDebug("Initializing CUDA context");
    m_Context = static_cast<CUcontext>(context);

    if (m_Context == nullptr)
    {
        throw KttException("The provided user CUDA context is not valid");
    }

    CheckError(cuCtxGetDevice(&m_Device), "cuCtxGetDevice");
}

CudaContext::~CudaContext()
{
    Logger::LogDebug("Releasing CUDA context");

    if (m_OwningContext)
    {
        CheckError(cuCtxDestroy(m_Context), "cuCtxDestroy");
    }
}

CUcontext CudaContext::GetContext() const
{
    return m_Context;
}

CUdevice CudaContext::GetDevice() const
{
    return m_Device;
}

} // namespace ktt

#endif // KTT_API_CUDA
