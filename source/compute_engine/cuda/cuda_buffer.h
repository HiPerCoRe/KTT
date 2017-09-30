#pragma once

#include <string>
#include <vector>

#include "cuda.h"
#include "cuda_utility.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"

namespace ktt
{

class CudaBuffer
{
public:
    explicit CudaBuffer(const size_t kernelArgumentId, const size_t bufferSize, const size_t elementSize, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType) :
        kernelArgumentId(kernelArgumentId),
        bufferSize(bufferSize),
        elementSize(elementSize),
        dataType(dataType),
        memoryLocation(memoryLocation),
        accessType(accessType)
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemAlloc(&deviceBuffer, bufferSize), "cuMemAlloc");
        }
        else
        {
            checkCudaError(cuMemAllocHost(&hostBufferRaw, bufferSize), "cuMemAllocHost");
            checkCudaError(cuMemHostGetDevicePointer(&hostBuffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
        }
    }

    ~CudaBuffer()
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemFree(deviceBuffer), "cuMemFree");
        }
        else
        {
            checkCudaError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
        }
    }

    void resize(const size_t newBufferSize)
    {
        if (bufferSize == newBufferSize)
        {
            return;
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemFree(deviceBuffer), "cuMemFree");
            checkCudaError(cuMemAlloc(&deviceBuffer, newBufferSize), "cuMemAlloc");
        }
        else
        {
            checkCudaError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
            checkCudaError(cuMemAllocHost(&hostBufferRaw, newBufferSize), "cuMemAllocHost");
            checkCudaError(cuMemHostGetDevicePointer(&hostBuffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
        }
        bufferSize = newBufferSize;
    }

    void uploadData(const void* source, const size_t dataSize)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize);
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemcpyHtoD(deviceBuffer, source, dataSize), "cuMemcpyHtoD");
        }
        else
        {
            checkCudaError(cuMemcpyHtoD(hostBuffer, source, dataSize), "cuMemcpyHtoD");
        }
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemcpyDtoH(destination, deviceBuffer, dataSize), "cuMemcpyDtoH");
        }
        else
        {
            checkCudaError(cuMemcpyDtoH(destination, hostBuffer, dataSize), "cuMemcpyDtoH");
        }
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

    ArgumentMemoryLocation getMemoryLocation() const
    {
        return memoryLocation;
    }

    ArgumentAccessType getAccessType() const
    {
        return accessType;
    }

    const CUdeviceptr* getBuffer() const
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            return &deviceBuffer;
        }
        else
        {
            return &hostBuffer;
        }
    }

    CUdeviceptr* getBuffer()
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            return &deviceBuffer;
        }
        else
        {
            return &hostBuffer;
        }
    }

private:
    size_t kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryLocation memoryLocation;
    ArgumentAccessType accessType;
    CUdeviceptr deviceBuffer;
    CUdeviceptr hostBuffer;
    void* hostBufferRaw;
};

} // namespace ktt
