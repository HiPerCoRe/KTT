#include <string>

#include <Api/KttException.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelArgumentManager::KernelArgumentManager() :
    m_IdGenerator(0)
{}

ArgumentId KernelArgumentManager::AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, void* data, const size_t dataSize)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetReferencedData(data, dataSize);
    return id;
}

ArgumentId KernelArgumentManager::AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const void* data, const size_t dataSize)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetOwnedData(data, dataSize);
    return id;
}

ArgumentId KernelArgumentManager::AddUserArgument(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, ArgumentMemoryType::Vector,
        ArgumentManagementType::User);
    auto& argument = GetArgument(id);
    argument.SetUserBuffer(dataSize);
    return id;
}

void KernelArgumentManager::RemoveArgument(const ArgumentId id)
{
    const size_t erasedCount = m_Arguments.erase(id);

    if (erasedCount == 0)
    {
        throw KttException("Attempting to remove argument with invalid id: " + std::to_string(id));
    }
}

const KernelArgument& KernelArgumentManager::GetArgument(const ArgumentId id) const
{
    if (!ContainsKey(m_Arguments, id))
    {
        throw KttException("Attempting to retrieve argument with invalid id: " + std::to_string(id));
    }

    return *m_Arguments.find(id)->second;
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
    m_Arguments[id] = std::move(argument);

    return id;
}

} // namespace ktt
