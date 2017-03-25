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
    ArgumentManager():
        argumentCount(0)
    {}

    // Core methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)
    {
        auto anyData = convertToAny(data);
        arguments.emplace_back(KernelArgument(anyData, argumentMemoryType));
        return argumentCount++;
    }

    template <typename T> void updateArgument(const size_t id, const std::vector<T>& data)
    {
        if (id >= argumentCount)
        {
            throw std::runtime_error(std::string("Invalid argument id: " + std::to_string(id)));
        }

        auto anyData = convertToAny(data);
        arguments.at(id).updateData(anyData);
    }

    // Getters
    size_t getArgumentCount() const
    {
        return argumentCount;
    }

    const KernelArgument getArgument(const size_t id) const
    {
        if (id >= argumentCount)
        {
            throw std::runtime_error(std::string("Invalid argument id: " + std::to_string(id)));
        }
        return arguments.at(id);
    }

private:
    // Attributes
    size_t argumentCount;
    std::vector<KernelArgument> arguments;

    // Helper methods
    template <typename T> std::vector<linb::any> convertToAny(const std::vector<T>& data) const
    {
        std::vector<linb::any> anyData;
        for (const auto element : data)
        {
            anyData.push_back(element);
        }
        return anyData;
    }
};

} // namespace ktt
