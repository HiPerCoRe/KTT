#include <stdexcept>
#include <string>
#include <kernel_argument/argument_manager.h>

namespace ktt
{

ArgumentManager::ArgumentManager() :
    nextArgumentId(0)
{}

ArgumentId ArgumentManager::addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType, const bool copyData)
{
    arguments.push_back(std::make_unique<KernelArgument>(nextArgumentId, data, numberOfElements, elementSizeInBytes, dataType, memoryLocation,
        accessType, uploadType, copyData));
    return nextArgumentId++;
}

ArgumentId ArgumentManager::addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
    const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
    const ArgumentUploadType uploadType)
{
    arguments.push_back(std::make_unique<KernelArgument>(nextArgumentId, data, numberOfElements, elementSizeInBytes, dataType, memoryLocation,
        accessType, uploadType));
    return nextArgumentId++;
}

ArgumentId ArgumentManager::addUserArgument(const size_t bufferSize, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType)
{
    arguments.push_back(std::make_unique<KernelArgument>(nextArgumentId, bufferSize, elementSize, dataType, memoryLocation, accessType));
    return nextArgumentId++;
}

void ArgumentManager::updateArgument(const ArgumentId id, void* data, const size_t numberOfElements)
{
    if (id >= nextArgumentId)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    arguments.at(id)->updateData(data, numberOfElements);
}

void ArgumentManager::updateArgument(const ArgumentId id, const void* data, const size_t numberOfElements)
{
    if (id >= nextArgumentId)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    arguments.at(id)->updateData(data, numberOfElements);
}

void ArgumentManager::setPersistentFlag(const ArgumentId id, const bool flag)
{
    if (id >= nextArgumentId)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    if (arguments.at(id)->getUploadType() != ArgumentUploadType::Vector)
    {
        throw std::runtime_error("Non-vector kernel arguments cannot be persistent");
    }
    arguments.at(id)->setPersistentFlag(flag);
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
    return *arguments.at(id);
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
        result.push_back(arguments.at(id).get());
    }

    return result;
}

} // namespace ktt