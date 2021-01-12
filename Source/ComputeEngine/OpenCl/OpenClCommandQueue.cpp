#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

OpenClCommandQueue::OpenClCommandQueue(const QueueId id, const OpenClContext& context) :
    m_Context(context.GetContext()),
    m_Id(id),
    m_OwningQueue(true)
{
    cl_int result;
#ifdef CL_VERSION_2_0
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    m_Queue = clCreateCommandQueueWithProperties(m_Context, context.GetDevice(), properties, &result);
    CheckError(result, "clCreateCommandQueueWithProperties");
#else
    m_Queue = clCreateCommandQueue(m_Context, context.GetDevice(), CL_QUEUE_PROFILING_ENABLE, &result);
    CheckError(result, "clCreateCommandQueue");
#endif
}

OpenClCommandQueue::OpenClCommandQueue(const QueueId id, const OpenClContext& context, ComputeQueue queue) :
    m_Context(context.GetContext()),
    m_Id(id),
    m_OwningQueue(false)
{
    m_Queue = static_cast<cl_command_queue>(queue);

    if (m_Queue == nullptr)
    {
        throw KttException("The provided user OpenCL queue is not valid");
    }

    cl_command_queue_properties properties;
    CheckError(clGetCommandQueueInfo(m_Queue, CL_QUEUE_PROPERTIES, sizeof(properties), &properties, nullptr),
        "clGetCommandQueueInfo");

    if (!(properties & CL_QUEUE_PROFILING_ENABLE))
    {
        throw KttException("The provided user OpenCL queue does not have profiling enabled");
    }
}

OpenClCommandQueue::~OpenClCommandQueue()
{
    if (m_OwningQueue)
    {
        CheckError(clReleaseCommandQueue(m_Queue), "clReleaseCommandQueue");
    }
}

cl_command_queue OpenClCommandQueue::GetQueue() const
{
    return m_Queue;
}

cl_context OpenClCommandQueue::GetContext() const
{
    return m_Context;
}

QueueId OpenClCommandQueue::GetId() const
{
    return m_Id;
}

} // namespace ktt

#endif // KTT_API_OPENCL
