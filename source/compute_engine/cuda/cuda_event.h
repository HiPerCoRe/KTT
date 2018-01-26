#pragma once

#include <string>
#include "cuda.h"
#include "cuda_utility.h"
#include "ktt_types.h"

namespace ktt
{

class CudaEvent
{
public:
    CudaEvent(const EventId id, const std::string& kernelName) :
        id(id),
        kernelName(kernelName)
    {
        checkCudaError(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
    }

    ~CudaEvent()
    {
        checkCudaError(cuEventDestroy(event), "cuEventDestroy");
    }

    EventId getId() const
    {
        return id;
    }

    std::string getKernelName() const
    {
        return kernelName;
    }

    CUevent getEvent() const
    {
        return event;
    }

private:
    EventId id;
    std::string kernelName;
    CUevent event;
};

} // namespace ktt
