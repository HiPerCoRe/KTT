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
        overhead(0),
        validFlag(validFlag),
        releaseFlag(false)
    {}

    OpenclEvent(const EventId id, const std::string& kernelName, const uint64_t kernelLaunchOverhead) :
        id(id),
        kernelName(kernelName),
        overhead(kernelLaunchOverhead),
        validFlag(true),
        releaseFlag(false)
    {}

    ~OpenclEvent()
    {
        if (releaseFlag)
        {
            checkOpenclError(clReleaseEvent(event), "clReleaseEvent");
        }
    }

    EventId getId() const
    {
        return id;
    }

    std::string getKernelName() const
    {
        return kernelName;
    }
    
    uint64_t getOverhead() const
    {
        return overhead;
    }

    bool isValid() const
    {
        return validFlag;
    }

    cl_event* getEvent()
    {
        return &event;
    }

    cl_ulong getEventCommandDuration() const
    {
        cl_ulong start;
        cl_ulong end;
        checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr), "clGetEventProfilingInfo");
        checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr), "clGetEventProfilingInfo");

        return end - start;
    }

    void setReleaseFlag()
    {
        releaseFlag = true;
    }

private:
    EventId id;
    std::string kernelName;
    uint64_t overhead;
    bool validFlag;
    bool releaseFlag;
    cl_event event;
};

} // namespace ktt
