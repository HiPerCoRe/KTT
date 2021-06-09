#pragma once

#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <KttTypes.h>

namespace ktt
{

class OpenClContext;

class OpenClCommandQueue
{
public:
    explicit OpenClCommandQueue(const QueueId id, const OpenClContext& context);
    explicit OpenClCommandQueue(const QueueId id, const OpenClContext& context, ComputeQueue queue);
    ~OpenClCommandQueue();

    void Synchronize() const;

    cl_command_queue GetQueue() const;
    cl_context GetContext() const;
    QueueId GetId() const;

private:
    cl_command_queue m_Queue;
    cl_context m_Context;
    QueueId m_Id;
    bool m_OwningQueue;
};

} // namespace ktt

#endif // KTT_API_OPENCL
