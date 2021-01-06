#include <Api/ComputeApiInitializer.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

ComputeApiInitializer::ComputeApiInitializer(ComputeContext context, const std::vector<ComputeQueue>& queues) :
    m_Context(context),
    m_Queues(queues)
{
    if (queues.empty())
    {
        throw KttException("No queues provided for compute API initializer");
    }
}

ComputeContext ComputeApiInitializer::GetContext() const
{
    return m_Context;
}

const std::vector<ComputeQueue>& ComputeApiInitializer::GetQueues() const
{
    return m_Queues;
}

} // namespace ktt
