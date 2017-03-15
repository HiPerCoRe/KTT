#pragma once

#include <vector>

#include "../../libraries/any.hpp"
#include "../enums/argument_memory_type.h"
#include "../enums/argument_quantity.h"

namespace ktt
{

using linb::any;
using linb::any_cast;

class KernelArgument
{
public:
    explicit KernelArgument(const std::vector<any>& data, const ArgumentMemoryType& argumentMemoryType):
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

    std::vector<any> getData() const
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

    template <typename T> bool isTypeOf() const
    {
        return data.at(0).type() == typeid(T);
    }

    template <typename T> std::vector<T> getDataTyped() const
    {
        if (!isTypeOf<T>())
        {
            throw std::runtime_error("Invalid argument data type");
        }

        std::vector<T> result;
        for (const auto& element : data)
        {
            result.push_back(any_cast<T>(element));
        }

        return result;
    }

private:
    std::vector<any> data;
    ArgumentMemoryType argumentMemoryType;
    ArgumentQuantity argumentQuantity;
};

} // namespace ktt
