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
        buffer(0),
        hostBufferRaw(nullptr),
        zeroCopy(zeroCopy)
    {
        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemAllocManaged(&buffer, bufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemAlloc(&buffer, bufferSize), "cuMemAlloc");
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
            checkCUDAError(cuMemHostGetDevicePointer(&buffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
        }
    }

    ~CUDABuffer()
    {
        if (memoryLocation == ArgumentMemoryLocation::Unified || memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemFree(buffer), "cuMemFree");
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
            if (memoryLocation == ArgumentMemoryLocation::Unified)
            {
                checkCUDAError(cuMemFree(buffer), "cuMemFree");
                checkCUDAError(cuMemAllocManaged(&buffer, newBufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
            }
            else if (memoryLocation == ArgumentMemoryLocation::Device)
            {
                checkCUDAError(cuMemFree(buffer), "cuMemFree");
                checkCUDAError(cuMemAlloc(&buffer, newBufferSize), "cuMemAlloc");
            }
            else
            {
                checkCUDAError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
                checkCUDAError(cuMemAllocHost(&hostBufferRaw, newBufferSize), "cuMemAllocHost");
                checkCUDAError(cuMemHostGetDevicePointer(&buffer, hostBufferRaw, 0), "cuMemHostGetDevicePointer");
            }
        }
        else
        {
            if (memoryLocation == ArgumentMemoryLocation::Unified)
            {
                CUdeviceptr newBuffer;
                checkCUDAError(cuMemAllocManaged(&newBuffer, newBufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
                checkCUDAError(cuMemcpy(newBuffer, buffer, std::min(bufferSize, newBufferSize)), "cuMemcpy");
                checkCUDAError(cuMemFree(buffer), "cuMemFree");
                buffer = newBuffer;
            }
            else if (memoryLocation == ArgumentMemoryLocation::Device)
            {
                CUdeviceptr newBuffer;
                checkCUDAError(cuMemAlloc(&newBuffer, newBufferSize), "cuMemAlloc");
                checkCUDAError(cuMemcpyDtoD(newBuffer, buffer, std::min(bufferSize, newBufferSize)), "cuMemcpyDtoD");
                checkCUDAError(cuMemFree(buffer), "cuMemFree");
                buffer = newBuffer;
            }
            else
            {
                void* newHostBufferRaw = nullptr;
                CUdeviceptr newBuffer;
                checkCUDAError(cuMemAllocHost(&newHostBufferRaw, newBufferSize), "cuMemAllocHost");
                checkCUDAError(cuMemHostGetDevicePointer(&newBuffer, newHostBufferRaw, 0), "cuMemHostGetDevicePointer");
                checkCUDAError(cuMemcpyDtoD(newBuffer, buffer, std::min(bufferSize, newBufferSize)), "cuMemcpyDtoD");
                checkCUDAError(cuMemFreeHost(hostBufferRaw), "cuMemFreeHost");
                hostBufferRaw = newHostBufferRaw;
                buffer = newBuffer;
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

        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemcpy(buffer, reinterpret_cast<CUdeviceptr>(source), dataSize), "cuMemcpy");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyHtoD(buffer, source, dataSize), "cuMemcpyHtoD");
        }
        else
        {
            checkCUDAError(cuMemcpyHtoD(buffer, source, dataSize), "cuMemcpyHtoD");
        }
    }

    void uploadData(CUstream stream, const void* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize, false);
        }

        checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");

        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemcpyAsync(buffer, reinterpret_cast<CUdeviceptr>(source), dataSize, stream), "cuMemcpyAsync");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyHtoDAsync(buffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyHtoDAsync(buffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
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

        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemcpyAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyAsync");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoDAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoDAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
        }

        checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemcpy(reinterpret_cast<CUdeviceptr>(destination), buffer, dataSize), "cuMemcpy");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoH(destination, buffer, dataSize), "cuMemcpyDtoH");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoH(destination, buffer, dataSize), "cuMemcpyDtoH");
        }
    }

    void downloadData(CUstream stream, void* destination, const size_t dataSize, CUevent startEvent, CUevent endEvent) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");

        if (memoryLocation == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(destination), buffer, dataSize, stream), "cuMemcpyAsync");
        }
        else if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            checkCUDAError(cuMemcpyDtoHAsync(destination, buffer, dataSize, stream), "cuMemcpyDtoHAsync");
        }
        else
        {
            checkCUDAError(cuMemcpyDtoHAsync(destination, buffer, dataSize, stream), "cuMemcpyDtoHAsync");
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
        return &buffer;
    }

    CUdeviceptr* getBuffer()
    {
        return &buffer;
    }

private:
    ArgumentId kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryLocation memoryLocation;
    ArgumentAccessType accessType;
    CUdeviceptr buffer;
    void* hostBufferRaw;
    bool zeroCopy;
};

} // namespace ktt
