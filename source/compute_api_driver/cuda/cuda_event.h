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
        checkCudaError(cuEventCreate(&event, CU_EVENT_DEFAULT), std::string("cuEventCreate"));
    }

    ~CudaEvent()
    {
        checkCudaError(cuEventDestroy(event), std::string("cuEventDestroy"));
    }

    CUevent getEvent() const
    {
        return event;
    }

private:
    CUevent event;
};

} // namespace ktt
