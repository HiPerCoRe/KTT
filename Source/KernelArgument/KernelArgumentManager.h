#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <KernelArgument/KernelArgument.h>
#include <Utility/IdGenerator.h>

namespace ktt
{

class KernelArgumentManager
{
public:
    KernelArgumentManager();

    ArgumentId AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, void* data, const size_t dataSize, const ArgumentId& customId = "");
    ArgumentId AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const void* data, const size_t dataSize, const ArgumentId& customId = "",
        const std::string& symbolName = "");
    ArgumentId AddArgumentWithOwnedDataFromFile(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const std::string& file, const ArgumentId& customId = "");
    ArgumentId AddArgumentWithOwnedDataFromGenerator(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const std::string& generatorFunction, const size_t dataSize,
        const ArgumentId& customId = "");
    ArgumentId AddUserArgument(const size_t elementSize, const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation,
        const ArgumentAccessType accessType, const size_t dataSize, const ArgumentId& customId = "");
    void RemoveArgument(const ArgumentId& id);

    const KernelArgument& GetArgument(const ArgumentId& id) const;
    KernelArgument& GetArgument(const ArgumentId& id);
    std::vector<KernelArgument*> GetArguments(const std::vector<ArgumentId>& ids);
    void SaveArgument(const ArgumentId& id, const std::string& file) const;

private:
    IdGenerator<uint64_t> m_IdGenerator;
    std::vector<std::unique_ptr<KernelArgument>> m_Arguments;

    ArgumentId AddArgument(const size_t elementSize, const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation,
        const ArgumentAccessType accessType, const ArgumentMemoryType memoryType, const ArgumentManagementType managementType,
        const ArgumentId& customId = "", const std::string& symbolName = "");
};

} // namespace ktt
