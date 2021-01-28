#pragma once

#include <cstdint>
#include <memory>
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
        void* data, const uint64_t numberOfElements);
    ArgumentId AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const void* data, const uint64_t numberOfElements);
    ArgumentId AddUserArgument(const size_t elementSize, const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation,
        const ArgumentAccessType accessType, const ArgumentMemoryType memoryType, const uint64_t numberOfElements);

    const KernelArgument& GetArgument(const ArgumentId id) const;
    KernelArgument& GetArgument(const ArgumentId id);
    std::vector<KernelArgument*> GetArguments(const std::vector<ArgumentId>& ids);

private:
    IdGenerator<ArgumentId> m_IdGenerator;
    std::vector<std::unique_ptr<KernelArgument>> m_Arguments;

    ArgumentId AddArgument(const size_t elementSize, const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation,
        const ArgumentAccessType accessType, const ArgumentMemoryType memoryType);
};

} // namespace ktt
