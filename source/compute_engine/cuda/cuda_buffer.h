#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include "cuda.h"
#include "cuda_utility.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class CudaBuffer
{
public:
    explicit CudaBuffer(KernelArgument& kernelArgument, const bool zeroCopy) :
        kernelArgumentId(kernelArgument.getId()),
        bufferSize(kernelArgument.getDataSizeInBytes()),
        elementSize(kernelArgument.getElementSizeInBytes()),
        dataType(kernelArgument.getDataType()),
        memoryLocation(kernelArgument.getMemoryLocation()),
        accessType(kernelArgument.getAccessType()),
        hostBufferRaw(nullptr),
        zeroCopy(zeroCopy)
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemAlloc(&deviceBuffer, bufferSize), "cuMemAlloc");
        }
        else
        {
            if (zeroCopy)
            {
                hostBufferRaw = kernelArgument.getData();
                checkCudaError(cuMemHostRegister(hostBufferRaw, bufferSize, CU_MEMHOSTREGISTER_DEVICEMAP), "cuMemHostRegister");
            }
            else
            {
                checkCudaError(cuMemAllocHost(&hostBufferRaw, bufferSize), "cuMemAllocHost");
            }
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
            if (zeroCopy)
            {
                checkCudaError(cuMemHostUnregister(hostBufferRaw), "cuMemHostUnregister");
            }
            else
            {
                checkCudaError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
            }
        }
    }

    void resize(const size_t newBufferSize)
    {
        if (zeroCopy)
        {
            throw std::runtime_error("Cannot resize registered host buffer");
        }

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

    void uploadData(CUstream stream, const void* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize);
        }

        checkCudaError(cuEventRecord(startEvent, stream), "cuEventRecord");
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemcpyHtoDAsync(deviceBuffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
        }
        else
        {
            checkCudaError(cuMemcpyHtoDAsync(hostBuffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
        }
        checkCudaError(cuEventRecord(endEvent, stream), "cuEventRecord");
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

    void downloadData(CUstream stream, void* destination, const size_t dataSize, CUevent startEvent, CUevent endEvent) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        checkCudaError(cuEventRecord(startEvent, stream), "cuEventRecord");
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCudaError(cuMemcpyDtoHAsync(destination, deviceBuffer, dataSize, stream), "cuMemcpyDtoHAsync");
        }
        else
        {
            checkCudaError(cuMemcpyDtoHAsync(destination, hostBuffer, dataSize, stream), "cuMemcpyDtoHAsync");
        }
        checkCudaError(cuEventRecord(endEvent, stream), "cuEventRecord");
    }

    ArgumentId getKernelArgumentId() const
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
    ArgumentId kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryLocation memoryLocation;
    ArgumentAccessType accessType;
    CUdeviceptr deviceBuffer;
    CUdeviceptr hostBuffer;
    void* hostBufferRaw;
    bool zeroCopy;
};

} // namespace ktt
