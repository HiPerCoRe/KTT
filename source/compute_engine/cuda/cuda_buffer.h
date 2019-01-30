#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda.h>
#include <compute_engine/cuda/cuda_utility.h>
#include <enum/argument_access_type.h>
#include <enum/argument_data_type.h>
#include <enum/argument_memory_location.h>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

class CUDABuffer
{
public:
    explicit CUDABuffer(KernelArgument& kernelArgument, const bool zeroCopy) :
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
            checkCUDAError(cuMemAlloc(&deviceBuffer, bufferSize), "cuMemAlloc");
        }
        else
        {
            if (zeroCopy)
            {
                hostBufferRaw = kernelArgument.getData();
                checkCUDAError(cuMemHostRegister(hostBufferRaw, bufferSize, CU_MEMHOSTREGISTER_DEVICEMAP), "cuMemHostRegister");
            }
            else
            {
                checkCUDAError(cuMemAllocHost(&hostBufferRaw, bufferSize), "cuMemAllocHost");
            }
            checkCUDAError(cuMemHostGetDevicePointer(&hostBuffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
        }
    }

    ~CUDABuffer()
    {
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemFree(deviceBuffer), "cuMemFree");
        }
        else
        {
            if (zeroCopy)
            {
                checkCUDAError(cuMemHostUnregister(hostBufferRaw), "cuMemHostUnregister");
            }
            else
            {
                checkCUDAError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
            }
        }
    }

    void resize(const size_t newBufferSize, const bool preserveData)
    {
        if (zeroCopy)
        {
            throw std::runtime_error("Cannot resize registered host buffer");
        }

        if (bufferSize == newBufferSize)
        {
            return;
        }

        if (!preserveData)
        {
            if (memoryLocation == ArgumentMemoryLocation::Device)
            {
                checkCUDAError(cuMemFree(deviceBuffer), "cuMemFree");
                checkCUDAError(cuMemAlloc(&deviceBuffer, newBufferSize), "cuMemAlloc");
            }
            else
            {
                checkCUDAError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
                checkCUDAError(cuMemAllocHost(&hostBufferRaw, newBufferSize), "cuMemAllocHost");
                checkCUDAError(cuMemHostGetDevicePointer(&hostBuffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
            }
        }
        else
        {
            if (memoryLocation == ArgumentMemoryLocation::Device)
            {
                CUdeviceptr newDeviceBuffer;
                checkCUDAError(cuMemAlloc(&newDeviceBuffer, newBufferSize), "cuMemAlloc");
                checkCUDAError(cuMemcpyDtoD(newDeviceBuffer, deviceBuffer, std::min(bufferSize, newBufferSize)), "cuMemcpyDtoD");
                checkCUDAError(cuMemFree(deviceBuffer), "cuMemFree");
                deviceBuffer = newDeviceBuffer;
            }
            else
            {
                void* newHostBufferRaw = nullptr;
                CUdeviceptr newHostBuffer;
                checkCUDAError(cuMemAllocHost(&newHostBufferRaw, newBufferSize), "cuMemAllocHost");
                checkCUDAError(cuMemHostGetDevicePointer(&newHostBuffer, newHostBufferRaw, 0), "cuMemHostGetDevicePointer");
                checkCUDAError(cuMemcpyDtoD(newHostBuffer, hostBuffer, std::min(bufferSize, newBufferSize)), "cuMemcpyDtoD");
                checkCUDAError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
                hostBufferRaw = newHostBufferRaw;
                hostBuffer = newHostBuffer;
            }
        }

        bufferSize = newBufferSize;
    }

    void uploadData(const void* source, const size_t dataSize)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize, false);
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyHtoD(deviceBuffer, source, dataSize), "cuMemcpyHtoD");
        }
        else
        {
            checkCUDAError(cuMemcpyHtoD(hostBuffer, source, dataSize), "cuMemcpyHtoD");
        }
    }

    void uploadData(CUstream stream, const void* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize, false);
        }

        checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyHtoDAsync(deviceBuffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyHtoDAsync(hostBuffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
        }
        checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
    }

    void uploadData(CUstream stream, const CUDABuffer* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize, false);
        }

        checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoDAsync(deviceBuffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoDAsync(hostBuffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
        }
        checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoH(destination, deviceBuffer, dataSize), "cuMemcpyDtoH");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoH(destination, hostBuffer, dataSize), "cuMemcpyDtoH");
        }
    }

    void downloadData(CUstream stream, void* destination, const size_t dataSize, CUevent startEvent, CUevent endEvent) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");
        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoHAsync(destination, deviceBuffer, dataSize, stream), "cuMemcpyDtoHAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoHAsync(destination, hostBuffer, dataSize, stream), "cuMemcpyDtoHAsync");
        }
        checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
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
