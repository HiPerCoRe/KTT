#ifdef KTT_PLATFORM_CUDA

#include <algorithm>
#include <stdexcept>
#include <compute_engine/cuda/cuda_buffer.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

CUDABuffer::CUDABuffer(KernelArgument& kernelArgument, const bool zeroCopy) :
    kernelArgument(kernelArgument),
    bufferSize(kernelArgument.getDataSizeInBytes()),
    buffer(0),
    hostBufferRaw(nullptr),
    zeroCopy(zeroCopy),
    userOwned(false)
{
    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemAllocManaged(&buffer, bufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
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

CUDABuffer::CUDABuffer(UserBuffer userBuffer, KernelArgument& kernelArgument) :
    kernelArgument(kernelArgument),
    bufferSize(kernelArgument.getDataSizeInBytes()),
    hostBufferRaw(nullptr),
    zeroCopy(false),
    userOwned(true)
{
    if (userBuffer == nullptr)
    {
        throw std::runtime_error("The provided user CUDA buffer is not valid");
    }

    buffer = reinterpret_cast<CUdeviceptr>(userBuffer);
}

CUDABuffer::~CUDABuffer()
{
    if (userOwned)
    {
        return;
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified || getMemoryLocation() == ArgumentMemoryLocation::Device)
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

void CUDABuffer::resize(const size_t newBufferSize, const bool preserveData)
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
        if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
        {
            checkCUDAError(cuMemFree(buffer), "cuMemFree");
            checkCUDAError(cuMemAllocManaged(&buffer, newBufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
        }
        else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
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
        if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
        {
            CUdeviceptr newBuffer;
            checkCUDAError(cuMemAllocManaged(&newBuffer, newBufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
            checkCUDAError(cuMemcpy(newBuffer, buffer, std::min(bufferSize, newBufferSize)), "cuMemcpy");
            checkCUDAError(cuMemFree(buffer), "cuMemFree");
            buffer = newBuffer;
        }
        else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
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

void CUDABuffer::uploadData(const void* source, const size_t dataSize)
{
    if (bufferSize < dataSize)
    {
        resize(dataSize, false);
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemcpy(buffer, reinterpret_cast<CUdeviceptr>(source), dataSize), "cuMemcpy");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        checkCUDAError(cuMemcpyHtoD(buffer, source, dataSize), "cuMemcpyHtoD");
    }
    else
    {
        checkCUDAError(cuMemcpyHtoD(buffer, source, dataSize), "cuMemcpyHtoD");
    }
}

void CUDABuffer::uploadData(CUstream stream, const void* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
{
    if (bufferSize < dataSize)
    {
        resize(dataSize, false);
    }

    checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemcpyAsync(buffer, reinterpret_cast<CUdeviceptr>(source), dataSize, stream), "cuMemcpyAsync");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        checkCUDAError(cuMemcpyHtoDAsync(buffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
    }
    else
    {
        checkCUDAError(cuMemcpyHtoDAsync(buffer, source, dataSize, stream), "cuMemcpyHtoDAsync");
    }

    checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
}

void CUDABuffer::uploadData(CUstream stream, const CUDABuffer* source, const size_t dataSize, CUevent startEvent, CUevent endEvent)
{
    if (bufferSize < dataSize)
    {
        resize(dataSize, false);
    }

    checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemcpyAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyAsync");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        checkCUDAError(cuMemcpyDtoDAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
    }
    else
    {
        checkCUDAError(cuMemcpyDtoDAsync(buffer, *source->getBuffer(), dataSize, stream), "cuMemcpyDtoDAsync");
    }

    checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
}

void CUDABuffer::downloadData(void* destination, const size_t dataSize) const
{
    if (bufferSize < dataSize)
    {
        throw std::runtime_error("Size of data to download is higher than size of buffer");
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemcpy(reinterpret_cast<CUdeviceptr>(destination), buffer, dataSize), "cuMemcpy");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        checkCUDAError(cuMemcpyDtoH(destination, buffer, dataSize), "cuMemcpyDtoH");
    }
    else
    {
        checkCUDAError(cuMemcpyDtoH(destination, buffer, dataSize), "cuMemcpyDtoH");
    }
}

void CUDABuffer::downloadData(CUstream stream, void* destination, const size_t dataSize, CUevent startEvent, CUevent endEvent) const
{
    if (bufferSize < dataSize)
    {
        throw std::runtime_error("Size of data to download is higher than size of buffer");
    }

    checkCUDAError(cuEventRecord(startEvent, stream), "cuEventRecord");

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        checkCUDAError(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(destination), buffer, dataSize, stream), "cuMemcpyAsync");
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        checkCUDAError(cuMemcpyDtoHAsync(destination, buffer, dataSize, stream), "cuMemcpyDtoHAsync");
    }
    else
    {
        checkCUDAError(cuMemcpyDtoHAsync(destination, buffer, dataSize, stream), "cuMemcpyDtoHAsync");
    }

    checkCUDAError(cuEventRecord(endEvent, stream), "cuEventRecord");
}

ArgumentId CUDABuffer::getKernelArgumentId() const
{
    return kernelArgument.getId();
}

size_t CUDABuffer::getBufferSize() const
{
    return bufferSize;
}

size_t CUDABuffer::getElementSize() const
{
    return kernelArgument.getElementSizeInBytes();
}

ArgumentDataType CUDABuffer::getDataType() const
{
    return kernelArgument.getDataType();
}

ArgumentMemoryLocation CUDABuffer::getMemoryLocation() const
{
    return kernelArgument.getMemoryLocation();
}

ArgumentAccessType CUDABuffer::getAccessType() const
{
    return kernelArgument.getAccessType();
}

const CUdeviceptr* CUDABuffer::getBuffer() const
{
    return &buffer;
}

CUdeviceptr* CUDABuffer::getBuffer()
{
    return &buffer;
}

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
