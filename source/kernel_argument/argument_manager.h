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
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType,
        const ArgumentQuantity& argumentQuantity)
    {
        arguments.emplace_back(KernelArgument(argumentCount, data, argumentMemoryType, argumentQuantity));
        return argumentCount++;
    }

    template <typename T> void updateArgument(const size_t id, const std::vector<T>& data, const ArgumentQuantity& argumentQuantity)
    {
        arguments.at(id).updateData(data, argumentQuantity);
    }

    // Getters
    size_t getArgumentCount() const;
    const KernelArgument getArgument(const size_t id) const;

private:
    // Attributes
    size_t argumentCount;
    std::vector<KernelArgument> arguments;
};

} // namespace ktt
