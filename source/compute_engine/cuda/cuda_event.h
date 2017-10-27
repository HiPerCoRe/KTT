#pragma once

#include "cuda.h"
#include "cuda_utility.h"

namespace ktt
{

class CudaEvent
{
public:
    CudaEvent()
    {
        checkCudaError(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
    }

    ~CudaEvent()
    {
        checkCudaError(cuEventDestroy(event), "cuEventDestroy");
    }

    CUevent getEvent() const
    {
        return event;
    }

private:
    CUevent event;
};

} // namespace ktt
