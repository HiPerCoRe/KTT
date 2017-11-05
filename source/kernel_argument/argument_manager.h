#pragma once

#include <memory>
#include <vector>
#include "kernel_argument.h"

namespace ktt
{

class ArgumentManager
{
public:
    // Constructor
    ArgumentManager();

    // Core methods
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType,
        const bool copyData);
    void updateArgument(const ArgumentId id, const void* data, const size_t numberOfElements);

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
