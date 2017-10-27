#include <string>
#include "argument_manager.h"

namespace ktt
{

ArgumentManager::ArgumentManager(const RunMode& runMode) :
    nextArgumentId(0),
    runMode(runMode)
{}

ArgumentId ArgumentManager::addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType)
{
    if (runMode == RunMode::Tuning)
    {
        arguments.emplace_back(nextArgumentId, data, numberOfElements, dataType, memoryLocation, accessType, uploadType);
    }
    else
    {
        arguments.emplace_back(nextArgumentId, data, numberOfElements, dataType, memoryLocation, accessType, uploadType,
            false);
    }
    return nextArgumentId++;
}

void ArgumentManager::updateArgument(const ArgumentId id, const void* data, const size_t numberOfElements)
{
    if (id >= nextArgumentId)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    arguments.at(static_cast<size_t>(id)).updateData(data, numberOfElements);
}

size_t ArgumentManager::getArgumentCount() const
{
    return arguments.size();
}

const KernelArgument& ArgumentManager::getArgument(const ArgumentId id) const
{
    if (id >= nextArgumentId)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    return arguments.at(static_cast<size_t>(id));
}

KernelArgument& ArgumentManager::getArgument(const ArgumentId id)
{
    return const_cast<KernelArgument&>(static_cast<const ArgumentManager*>(this)->getArgument(id));
}

std::vector<KernelArgument*> ArgumentManager::getArguments(const std::vector<ArgumentId>& argumentIds)
{
    std::vector<KernelArgument*> result;

    for (const auto id : argumentIds)
    {
        if (id >= nextArgumentId)
        {
            throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
        }
        result.push_back(&arguments.at(static_cast<size_t>(id)));
    }

    return result;
}

} // namespace ktt
