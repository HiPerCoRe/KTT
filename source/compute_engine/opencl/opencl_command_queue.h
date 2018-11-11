#pragma once

#include <vector>
#include <CL/cl.h>
#include <compute_engine/opencl/opencl_utility.h>
#include <ktt_types.h>

namespace ktt
{

class OpenCLCommandQueue
{
public:
    explicit OpenCLCommandQueue(const QueueId id, const cl_context context, const cl_device_id device) :
        id(id),
        context(context),
        device(device)
    {
        cl_int result;
        #ifdef CL_VERSION_2_0
        cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        queue = clCreateCommandQueueWithProperties(context, device, properties, &result);
        checkOpenCLError(result, "clCreateCommandQueueWithProperties");
        #else
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &result);
        checkOpenCLError(result, "clCreateCommandQueue");
        #endif
    }

    ~OpenCLCommandQueue()
    {
        checkOpenCLError(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
    }

    QueueId getId() const
    {
        return id;
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
    QueueId id;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
};

} // namespace ktt
