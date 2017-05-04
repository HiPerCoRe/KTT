#include <string>

#include "argument_manager.h"

namespace ktt
{

ArgumentManager::ArgumentManager() :
    argumentCount(0)
{}

void ArgumentManager::enableArgumentPrinting(const std::vector<size_t> argumentIds, const std::string& filePath,
    const ArgumentPrintCondition& argumentPrintCondition)
{
    for (const auto id : argumentIds)
    {
        if (id >= argumentCount)
        {
            throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
        }

        if (arguments.at(id).getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
        {
            throw std::runtime_error(std::string("Argument with id: ") + std::to_string(id) + " is read-only");
        }

        arguments.at(id).enablePrinting(filePath, argumentPrintCondition);
    }
}

size_t ArgumentManager::getArgumentCount() const
{
    return argumentCount;
}

const KernelArgument ArgumentManager::getArgument(const size_t id) const
{
    if (id >= argumentCount)
    {
        throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(id));
    }
    return arguments.at(id);
}

} // namespace ktt
