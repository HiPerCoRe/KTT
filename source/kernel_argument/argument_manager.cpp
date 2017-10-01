#include <string>

#include "argument_manager.h"

namespace ktt
{

ArgumentManager::ArgumentManager(const RunMode& runMode) :
    argumentCount(0),
    runMode(runMode)
{}

size_t ArgumentManager::addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType)
{
    if (runMode == RunMode::Tuning)
    {
        arguments.emplace_back(argumentCount, data, numberOfElements, dataType, memoryLocation, accessType, uploadType);
    }
    else
    {
        arguments.emplace_back(argumentCount, data, numberOfElements, dataType, memoryLocation, accessType, uploadType, false);
    }
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
