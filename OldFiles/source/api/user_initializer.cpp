#include <stdexcept>
#include <api/user_initializer.h>

namespace ktt
{

UserInitializer::UserInitializer(UserContext context, const std::vector<UserQueue>& queues) :
    context(context),
    queues(queues)
{
    if (queues.empty())
    {
        throw std::runtime_error("No queues provided inside user initializer");
    }
}

UserContext UserInitializer::getContext() const
{
    return context;
}

const std::vector<UserQueue>& UserInitializer::getQueues() const
{
    return queues;
}

} // namespace ktt
