#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <CL/cl.h>
#include <ktt_types.h>

namespace ktt
{

class OpenCLCommandQueue
{
public:
    explicit OpenCLCommandQueue(const QueueId id, const cl_context context, const cl_device_id device);
    explicit OpenCLCommandQueue(const QueueId id, const cl_context context, const cl_device_id device, UserQueue queue);
    ~OpenCLCommandQueue();

    cl_command_queue getQueue() const;
    cl_context getContext() const;
    cl_device_id getDevice() const;
    QueueId getId() const;

private:
    cl_command_queue queue;
    cl_context context;
    cl_device_id device;
    QueueId id;
    bool owningQueue;
};

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
