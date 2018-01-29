#pragma once

#include <cstdint>
#include <string>
#include "CL/cl.h"
#include "ktt_types.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclEvent
{
public:
    OpenclEvent(const EventId id, const bool validFlag) :
        id(id),
        kernelName(""),
        validFlag(validFlag),
        overhead(0)
    {}

    OpenclEvent(const EventId id, const std::string& kernelName, const uint64_t kernelLaunchOverhead) :
        id(id),
        kernelName(kernelName),
        validFlag(true),
        overhead(kernelLaunchOverhead)
    {}

    ~OpenclEvent()
    {
        checkOpenclError(clReleaseEvent(event), "clReleaseEvent");
    }

    EventId getId() const
    {
        return id;
    }

    std::string getKernelName() const
    {
        return kernelName;
    }

    cl_event* getEvent()
    {
        return &event;
    }

    bool isValid() const
    {
        return validFlag;
    }
    
    uint64_t getOverhead() const
    {
        return overhead;
    }

    cl_ulong getEventCommandDuration() const
    {
        cl_ulong start;
        cl_ulong end;
        checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr), "clGetEventProfilingInfo");
        checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr), "clGetEventProfilingInfo");

        return end - start;
    }

private:
    EventId id;
    std::string kernelName;
    cl_event event;
    bool validFlag;
    uint64_t overhead;
};

} // namespace ktt
