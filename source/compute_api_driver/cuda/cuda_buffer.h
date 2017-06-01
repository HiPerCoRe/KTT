#pragma once

#include <string>
#include <vector>

#include "cuda.h"
#include "cuda_utility.h"
#include "../../enum/argument_memory_type.h"

namespace ktt
{

class CudaBuffer
{
public:
    explicit CudaBuffer(const ArgumentMemoryType& type, const size_t size, const size_t kernelArgumentId) :
        type(type),
        size(size),
        kernelArgumentId(kernelArgumentId)
    {
        checkCudaError(cuMemAlloc(&buffer, size), std::string("cuMemAlloc"));
    }

    ~CudaBuffer()
    {
        checkCudaError(cuMemFree(buffer), std::string("cuMemFree"));
    }

    ArgumentMemoryType getType() const
    {
        return type;
    }

    size_t getSize() const
    {
        return size;
    }

    CUdeviceptr getBuffer() const
    {
        return buffer;
    }

    size_t getKernelArgumentId() const
    {
        return kernelArgumentId;
    }

private:
    ArgumentMemoryType type;
    size_t size;
    CUdeviceptr buffer;
    size_t kernelArgumentId;
};

} // namespace ktt
