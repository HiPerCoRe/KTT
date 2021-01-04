#ifdef KTT_PLATFORM_OPENCL

#include <stdexcept>
#include <compute_engine/opencl/opencl_command_queue.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

OpenCLCommandQueue::OpenCLCommandQueue(const QueueId id, const cl_context context, const cl_device_id device) :
    context(context),
    device(device),
    id(id),
    owningQueue(true)
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

OpenCLCommandQueue::OpenCLCommandQueue(const QueueId id, const cl_context context, const cl_device_id device, UserQueue queue) :
    context(context),
    device(device),
    id(id),
    owningQueue(false)
{
    this->queue = static_cast<cl_command_queue>(queue);

    if (this->queue == nullptr)
    {
        throw std::runtime_error("The provided user OpenCL queue is not valid");
    }

    cl_command_queue_properties properties;
    checkOpenCLError(clGetCommandQueueInfo(this->queue, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &properties, nullptr),
        "clGetCommandQueueInfo");

    if (!(properties & CL_QUEUE_PROFILING_ENABLE))
    {
        throw std::runtime_error("The provided user OpenCL queue does not have profiling enabled");
    }
}

OpenCLCommandQueue::~OpenCLCommandQueue()
{
    if (owningQueue)
    {
        checkOpenCLError(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
    }
}

cl_command_queue OpenCLCommandQueue::getQueue() const
{
    return queue;
}

cl_context OpenCLCommandQueue::getContext() const
{
    return context;
}

cl_device_id OpenCLCommandQueue::getDevice() const
{
    return device;
}

QueueId OpenCLCommandQueue::getId() const
{
    return id;
}

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
