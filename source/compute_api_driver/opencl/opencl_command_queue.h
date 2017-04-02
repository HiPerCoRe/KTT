#pragma once

#include <vector>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenCLCommandQueue
{
public:
    explicit OpenCLCommandQueue(const cl_context context, const cl_device_id device):
        context(context),
        device(device)
    {
        cl_int result;
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &result);
        checkOpenCLError(result, std::string("clCreateCommandQueue"));
    }

    ~OpenCLCommandQueue()
    {
        checkOpenCLError(clReleaseCommandQueue(queue), std::string("clReleaseCommandQueue"));
    }

    cl_context getContext() const
    {
        return context;
    }

    cl_device_id getDevice() const
    {
        return device;
    }

    cl_command_queue getQueue() const
    {
        return queue;
    }

private:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
};

} // namespace ktt
