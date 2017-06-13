#pragma once

#include <string>
#include <vector>

#include "cuda.h"
#include "cuda_utility.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_type.h"

namespace ktt
{

class CudaBuffer
{
public:
    explicit CudaBuffer(const size_t kernelArgumentId, const size_t bufferSize, const size_t elementSize, const ArgumentDataType& dataType,
        const ArgumentMemoryType& memoryType) :
        kernelArgumentId(kernelArgumentId),
        bufferSize(bufferSize),
        elementSize(elementSize),
        dataType(dataType),
        memoryType(memoryType)
    {
        checkCudaError(cuMemAlloc(&buffer, bufferSize), std::string("cuMemAlloc"));
    }

    ~CudaBuffer()
    {
        checkCudaError(cuMemFree(buffer), std::string("cuMemFree"));
    }

    void uploadData(const void* source, const size_t dataSize)
    {
        if (bufferSize != dataSize)
        {
            checkCudaError(cuMemFree(buffer), std::string("cuMemFree"));
            checkCudaError(cuMemAlloc(&buffer, dataSize), std::string("cuMemAlloc"));
            bufferSize = dataSize;
        }
        checkCudaError(cuMemcpyHtoD(buffer, source, dataSize), std::string("cuMemcpyHtoD"));
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }
        checkCudaError(cuMemcpyDtoH(destination, buffer, dataSize), std::string("cuMemcpyDtoH"));
    }

    size_t getKernelArgumentId() const
    {
        return kernelArgumentId;
    }

    size_t getBufferSize() const
    {
        return bufferSize;
    }

    size_t getElementSize() const
    {
        return elementSize;
    }

    ArgumentDataType getDataType() const
    {
        return dataType;
    }

    ArgumentMemoryType getMemoryType() const
    {
        return memoryType;
    }

    const CUdeviceptr* getBuffer() const
    {
        return &buffer;
    }

    CUdeviceptr* getBuffer()
    {
        return &buffer;
    }



private:
    size_t kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryType memoryType;
    CUdeviceptr buffer;
};

} // namespace ktt
