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
    size_t addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType, const ArgumentQuantity& argumentQuantity);
    void updateArgument(const size_t id, const void* data, const size_t numberOfElements);

    // Getters
    size_t getArgumentCount() const;
    const KernelArgument& getArgument(const size_t id) const;

private:
    // Attributes
    size_t argumentCount;
    std::vector<KernelArgument> arguments;
};

} // namespace ktt
