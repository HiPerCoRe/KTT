#pragma once

#include <vector>

#include "../enums/argument_memory_type.h"
#include "../enums/argument_quantity.h"

namespace ktt
{

template <typename T> class KernelArgument
{
public:
    explicit KernelArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType):
        data(data),
        argumentMemoryType(argumentMemoryType)
    {
        if (data.size() == 0)
        {
            throw std::runtime_error("Data provided for kernel argument is empty.");
        }
        else if (data.size() == 1)
        {
            argumentQuantity = ArgumentQuantity::Scalar;
        }
        else
        {
            argumentQuantity = ArgumentQuantity::Vector;
        }
    }

    std::vector<T> getData() const
    {
        return data;
    }

    ArgumentMemoryType getArgumentMemoryType() const
    {
        return argumentMemoryType;
    }

    ArgumentQuantity getArgumentQuantity() const
    {
        return argumentQuantity;
    }

private:
    std::vector<T> data;
    ArgumentMemoryType argumentMemoryType;
    ArgumentQuantity argumentQuantity;
};

} // namespace ktt
