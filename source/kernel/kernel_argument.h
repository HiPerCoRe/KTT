#pragma once

#include <vector>

#include "../enums/kernel_argument_quantity.h"

namespace ktt
{

template <typename T> class KernelArgument
{
public:
    explicit KernelArgument(const std::vector<T>& data, const KernelArgumentQuantity& kernelArgumentQuantity):
        data(data),
        kernelArgumentQuantity(kernelArgumentQuantity)
    {}

    std::vector<T> getData() const
    {
        return data;
    }

    KernelArgumentQuantity getKernelArgumentQuantity() const
    {
        return kernelArgumentQuantity;
    }

private:
    std::vector<T> data;
    KernelArgumentQuantity kernelArgumentQuantity;
};

} // namespace ktt
