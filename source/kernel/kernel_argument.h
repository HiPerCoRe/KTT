#pragma once

#include <vector>

#include "../enums/kernel_argument_type.h"

namespace ktt
{

template <typename T> class KernelArgument
{
public:
    explicit KernelArgument(const size_t index, const std::vector<T>& data, const KernelArgumentType& kernelArgumentType):
        index(index),
        data(data),
        kernelArgumentType(kernelArgumentType)
    {}
    
    size_t getIndex() const
    {
        return index;
    }

    std::vector<T> getData() const
    {
        return data;
    }

    KernelArgumentType getKernelArgumentType() const
    {
        return kernelArgumentType;
    }

private:
    size_t index;
    std::vector<T> data;
    KernelArgumentType kernelArgumentType;
};

} // namespace ktt
