#pragma once

#ifdef KTT_API_CUDA

#include <cuda.h>

#include <KttTypes.h>

namespace ktt
{

class CudaEvent
{
public:
    CudaEvent();
    ~CudaEvent();

    void WaitForFinish() const;
    CUevent GetEvent() const;

    static Nanoseconds GetDuration(const CudaEvent& start, const CudaEvent& end);

private:
    CUevent m_Event;
};

} // namespace ktt

#endif // KTT_API_CUDA
