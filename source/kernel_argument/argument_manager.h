#pragma once

#include <memory>
#include <vector>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

class ArgumentManager
{
public:
    // Constructor
    ArgumentManager();

    // Core methods
    ArgumentId addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType, const bool copyData);
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType);
    void updateArgument(const ArgumentId id, void* data, const size_t numberOfElements);
    void updateArgument(const ArgumentId id, const void* data, const size_t numberOfElements);
    void setPersistentFlag(const ArgumentId id, const bool flag);

    // Getters
    size_t getArgumentCount() const;
    const KernelArgument& getArgument(const ArgumentId id) const;
    KernelArgument& getArgument(const ArgumentId id);
    std::vector<KernelArgument*> getArguments(const std::vector<ArgumentId>& argumentIds);

private:
    // Attributes
    ArgumentId nextArgumentId;
    std::vector<KernelArgument> arguments;
};

} // namespace ktt
