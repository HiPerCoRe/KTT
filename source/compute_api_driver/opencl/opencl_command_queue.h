#pragma once

#include <vector>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclCommandQueue
{
public:
    explicit OpenclCommandQueue(const cl_context context, const cl_device_id device) :
        context(context),
        device(device)
    {
        cl_int result;
        #ifdef CL_VERSION_2_0
            cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
            queue = clCreateCommandQueueWithProperties(context, device, properties, &result);
            checkOpenclError(result, std::string("clCreateCommandQueueWithProperties"));
        #else
            queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &result);
            checkOpenclError(result, std::string("clCreateCommandQueue"));
        #endif
    }

    ~OpenclCommandQueue()
    {
        checkOpenclError(clReleaseCommandQueue(queue), std::string("clReleaseCommandQueue"));
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
