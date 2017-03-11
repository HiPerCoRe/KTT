#pragma once

#include <vector>

#include "../enums/kernel_argument_access_type.h"
#include "../enums/kernel_argument_quantity.h"

namespace ktt
{

template <typename T> class KernelArgument
{
public:
    explicit KernelArgument(const std::vector<T>& data, const KernelArgumentQuantity& kernelArgumentQuantity,
        const KernelArgumentAccessType& kernelArgumentAccessType):
        data(data),
        kernelArgumentQuantity(kernelArgumentQuantity),
        kernelArgumentAccessType(kernelArgumentAccessType)
    {}

    std::vector<T> getData() const
    {
        return data;
    }

    KernelArgumentQuantity getKernelArgumentQuantity() const
    {
        return kernelArgumentQuantity;
    }

    KernelArgumentAccessType getKernelArgumentAccessType() const
    {
        return kernelArgumentAccessType;
    }

private:
    std::vector<T> data;
    KernelArgumentQuantity kernelArgumentQuantity;
    KernelArgumentAccessType kernelArgumentAccessType;
};

} // namespace ktt
