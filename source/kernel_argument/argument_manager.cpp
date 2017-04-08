#include <string>

#include "argument_manager.h"

namespace ktt
{

ArgumentManager::ArgumentManager():
    argumentCount(0)
{}

size_t ArgumentManager::getArgumentCount() const
{
    return argumentCount;
}

const KernelArgument ArgumentManager::getArgument(const size_t id) const
{
    if (id >= argumentCount)
    {
        throw std::runtime_error(std::string("Invalid argument id: " + std::to_string(id)));
    }
    return arguments.at(id);
}

} // namespace ktt
