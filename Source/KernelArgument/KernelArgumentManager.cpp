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
    const ArgumentManagementType managementType, void* data, const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetReferencedData(data, numberOfElements);
    return id;
}

ArgumentId KernelArgumentManager::AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const void* data, const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetOwnedData(data, numberOfElements);
    return id;
}

ArgumentId KernelArgumentManager::AddUserArgument(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const uint64_t numberOfElements)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetUserBuffer(numberOfElements);
    return id;
}

const KernelArgument& KernelArgumentManager::GetArgument(const ArgumentId id) const
{
    if (id >= static_cast<uint64_t>(m_Arguments.size()))
    {
        throw KttException("Attempting to retrieve argument with invalid id: " + std::to_string(id));
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
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType)
{
    const auto id = m_IdGenerator.GenerateId();

    auto argument = std::make_unique<KernelArgument>(id, elementSize, dataType, memoryLocation, accessType, memoryType,
        managementType);
    m_Arguments.push_back(std::move(argument));

    return id;
}

} // namespace ktt
