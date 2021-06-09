#ifdef KTT_API_CUDA

#include <ComputeEngine/Cuda/CudaEvent.h>
#include <ComputeEngine/Cuda/CudaUtility.h>

namespace ktt
{

CudaEvent::CudaEvent()
{
    CheckError(cuEventCreate(&m_Event, CU_EVENT_DEFAULT), "cuEventCreate");
}

CudaEvent::~CudaEvent()
{
    CheckError(cuEventDestroy(m_Event), "cuEventDestroy");
}

void CudaEvent::WaitForFinish() const
{
    CheckError(cuEventSynchronize(m_Event), "cuEventSynchronize");
}

CUevent CudaEvent::GetEvent() const
{
    return m_Event;
}

Nanoseconds CudaEvent::GetDuration(const CudaEvent& start, const CudaEvent& end)
{
    float duration;
    CheckError(cuEventElapsedTime(&duration, start.GetEvent(), end.GetEvent()), "cuEventElapsedTime");

    duration *= 1'000'000.0f;
    return static_cast<Nanoseconds>(duration);
}

} // namespace ktt

#endif // KTT_API_CUDA
