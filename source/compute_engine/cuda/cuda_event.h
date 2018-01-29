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
    CudaEvent(const EventId id, const bool validFlag) :
        id(id),
        kernelName(""),
        validFlag(validFlag)
    {
        checkCudaError(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
    }

    CudaEvent(const EventId id, const std::string& kernelName) :
        id(id),
        kernelName(kernelName),
        validFlag(true)
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

    bool isValid() const
    {
        return validFlag;
    }

private:
    EventId id;
    std::string kernelName;
    CUevent event;
    bool validFlag;
};

} // namespace ktt
