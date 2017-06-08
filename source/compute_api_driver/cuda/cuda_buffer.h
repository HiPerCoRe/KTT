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

    void uploadData(const void* source, const size_t dataSize)
    {
        checkCudaError(cuMemcpyHtoD(buffer, source, dataSize), std::string("cuMemcpyHtoD"));
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        checkCudaError(cuMemcpyDtoH(destination, buffer, dataSize), std::string("cuMemcpyDtoH"));
    }

    ArgumentMemoryType getType() const
    {
        return type;
    }

    size_t getSize() const
    {
        return size;
    }

    const CUdeviceptr* getBuffer() const
    {
        return &buffer;
    }

    CUdeviceptr* getBuffer()
    {
        return &buffer;
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
