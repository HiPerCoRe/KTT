#pragma once

#include <memory>
#include <vector>

#include "kernel_argument.h"
#include "enum/run_mode.h"

namespace ktt
{

class ArgumentManager
{
public:
    // Constructor
    ArgumentManager(const RunMode& runMode);

    // Core methods
    size_t addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType);
    void updateArgument(const size_t id, const void* data, const size_t numberOfElements);

    // Getters
    size_t getArgumentCount() const;
    const KernelArgument& getArgument(const size_t id) const;

private:
    // Attributes
    size_t argumentCount;
    std::vector<KernelArgument> arguments;
    RunMode runMode;
};

} // namespace ktt
