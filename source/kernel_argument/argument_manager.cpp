#include <string>

#include "argument_manager.h"

namespace ktt
{

ArgumentManager::ArgumentManager() :
    argumentCount(0)
{}

size_t ArgumentManager::addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
    const ArgumentMemoryType& argumentMemoryType, const ArgumentQuantity& argumentQuantity)
{
    arguments.emplace_back(KernelArgument(argumentCount, data, numberOfElements, argumentDataType, argumentMemoryType, argumentQuantity));
    return argumentCount++;
}

void ArgumentManager::updateArgument(const size_t id, const void* data, const size_t numberOfElements)
{
    if (id >= argumentCount)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    arguments.at(id).updateData(data, numberOfElements);
}

size_t ArgumentManager::getArgumentCount() const
{
    return argumentCount;
}

const KernelArgument& ArgumentManager::getArgument(const size_t id) const
{
    if (id >= argumentCount)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    return arguments.at(id);
}

} // namespace ktt
