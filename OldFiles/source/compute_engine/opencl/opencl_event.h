#pragma once

#include <cstdint>
#include <string>
#include <CL/cl.h>
#include <api/kernel_compilation_data.h>
#include <compute_engine/opencl/opencl_utility.h>
#include <ktt_types.h>

namespace ktt
{

class OpenCLEvent
{
public:
    OpenCLEvent(const EventId id, const bool validFlag) :
        id(id),
        kernelName(""),
        overhead(0),
        validFlag(validFlag),
        releaseFlag(false)
    {}

    OpenCLEvent(const EventId id, const std::string& kernelName, const uint64_t kernelLaunchOverhead, const KernelCompilationData& compilationData) :
        id(id),
        kernelName(kernelName),
        overhead(kernelLaunchOverhead),
        compilationData(compilationData),
        validFlag(true),
        releaseFlag(false)
    {}

    ~OpenCLEvent()
    {
        if (releaseFlag)
        {
            checkOpenCLError(clReleaseEvent(event), "clReleaseEvent");
        }
    }

    EventId getId() const
    {
        return id;
    }

    const std::string& getKernelName() const
    {
        return kernelName;
    }
    
    uint64_t getOverhead() const
    {
        return overhead;
    }

    const KernelCompilationData& getCompilationData() const
    {
        return compilationData;
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
        checkOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr), "clGetEventProfilingInfo");
        checkOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr), "clGetEventProfilingInfo");

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
    KernelCompilationData compilationData;
    bool validFlag;
    bool releaseFlag;
    cl_event event;
};

} // namespace ktt
