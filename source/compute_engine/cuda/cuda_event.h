#pragma once

#include <cstdint>
#include <string>
#include <cuda.h>
#include <compute_engine/cuda/cuda_utility.h>
#include <ktt_types.h>

namespace ktt
{

class CUDAEvent
{
public:
    CUDAEvent(const EventId id, const bool validFlag) :
        id(id),
        kernelName(""),
        validFlag(validFlag),
        overhead(0)
    {
        checkCUDAError(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
    }

    CUDAEvent(const EventId id, const std::string& kernelName, const uint64_t kernelLaunchOverhead) :
        id(id),
        kernelName(kernelName),
        validFlag(true),
        overhead(kernelLaunchOverhead)
    {
        checkCUDAError(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
    }

    ~CUDAEvent()
    {
        checkCUDAError(cuEventDestroy(event), "cuEventDestroy");
    }

    EventId getId() const
    {
        return id;
    }

    const std::string& getKernelName() const
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

    uint64_t getOverhead() const
    {
        return overhead;
    }

private:
    EventId id;
    std::string kernelName;
    CUevent event;
    bool validFlag;
    uint64_t overhead;
};

} // namespace ktt
