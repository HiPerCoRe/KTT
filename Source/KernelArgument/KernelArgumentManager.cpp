#include <algorithm>
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
    const ArgumentManagementType managementType, const void* data, const size_t dataSize, const std::string& symbolName)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType, symbolName);
    auto& argument = GetArgument(id);
    argument.SetOwnedData(data, dataSize);
    return id;
}

ArgumentId KernelArgumentManager::AddArgumentWithOwnedDataFromFile(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const std::string& file)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetOwnedDataFromFile(file);
    return id;
}

ArgumentId KernelArgumentManager::AddArgumentWithOwnedDataFromGenerator(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const std::string& generatorFunction, const size_t dataSize)
{
    const auto id = AddArgument(elementSize, dataType, memoryLocation, accessType, memoryType, managementType);
    auto& argument = GetArgument(id);
    argument.SetOwnedDataFromGenerator(generatorFunction, dataSize);
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
    const size_t erasedCount = EraseIf(m_Arguments, [id](const auto& argument)
    {
        return argument->GetId() == id;
    });

    if (erasedCount == 0)
    {
        throw KttException("Attempting to remove argument with invalid id: " + std::to_string(id));
    }
}

const KernelArgument& KernelArgumentManager::GetArgument(const ArgumentId id) const
{
    auto iterator = std::find_if(m_Arguments.cbegin(), m_Arguments.cend(), [id](const auto& argument)
    {
        return argument->GetId() == id;
    });

    if (iterator == m_Arguments.cend())
    {
        throw KttException("Attempting to retrieve argument with invalid id: " + std::to_string(id));
    }

    return **iterator;
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

void KernelArgumentManager::SaveArgument(const ArgumentId id, const std::string& file) const
{
    GetArgument(id).SaveData(file);
}

ArgumentId KernelArgumentManager::AddArgument(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const std::string& symbolName)
{
    if (memoryType == ArgumentMemoryType::Symbol && !symbolName.empty())
    {
        const bool symbolNameExists = std::any_of(m_Arguments.cbegin(), m_Arguments.cend(), [&symbolName](const auto& argument)
        {
            return argument->GetSymbolName() == symbolName;
        });

        if (symbolNameExists)
        {
            throw KttException("Kernel argument with symbol name " + symbolName + " already exists");
        }
    }

    if (memoryType == ArgumentMemoryType::Vector && memoryLocation == ArgumentMemoryLocation::Undefined)
    {
        throw KttException("Vector kernel arguments must have properly defined memory location");
    }

    const auto id = m_IdGenerator.GenerateId();

    auto argument = std::make_unique<KernelArgument>(id, elementSize, dataType, memoryLocation, accessType, memoryType,
        managementType, symbolName);
    m_Arguments.push_back(std::move(argument));

    return id;
}

} // namespace ktt
