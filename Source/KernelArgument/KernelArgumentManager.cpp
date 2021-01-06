#include <string>

#include <KernelArgument/KernelArgumentManager.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelArgumentManager::KernelArgumentManager() :
    m_IdGenerator(0)
{}

ArgumentId KernelArgumentManager::AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    void* data, const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType);
    auto& argument = GetArgument(id);
    argument.SetReferencedData(data, numberOfElements);
    return id;
}

ArgumentId KernelArgumentManager::AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const void* data, const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType);
    auto& argument = GetArgument(id);
    argument.SetOwnedData(data, numberOfElements);
    return id;
}

ArgumentId KernelArgumentManager::AddUserArgument(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType);
    auto& argument = GetArgument(id);
    argument.SetUserBuffer(numberOfElements);
    return id;
}

const KernelArgument& KernelArgumentManager::GetArgument(const ArgumentId id) const
{
    if (id >= m_IdGenerator)
    {
        throw KttException(std::string("Attempting to retrieve argument with invalid id: ") + std::to_string(id));
    }

    return *m_Arguments[id];
}

KernelArgument& KernelArgumentManager::GetArgument(const ArgumentId id)
{
    return const_cast<KernelArgument&>(static_cast<const KernelArgumentManager*>(this)->GetArgument(id));
}

std::vector<KernelArgument*> KernelArgumentManager::GetArguments(const std::vector<ArgumentId>& ids)
{
    std::vector<KernelArgument*> result;

    for (const auto id : ids)
    {
        auto& argument = GetArgument(id);
        result.push_back(&argument);
    }

    return result;
}

ArgumentId KernelArgumentManager::AddArgument(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType)
{
    const auto id = m_IdGenerator;
    ++m_IdGenerator;

    auto argument = std::make_unique<KernelArgument>(id, elementSize, dataType, memoryLocation, accessType, memoryType);
    m_Arguments.push_back(std::move(argument));

    return id;
}

} // namespace ktt
