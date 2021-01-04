#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <cuda.h>
#include <kernel_argument/kernel_argument.h>
#include <ktt_types.h>

namespace ktt
{

class CUDABuffer
{
public:
    explicit CUDABuffer(KernelArgument& kernelArgument);
    explicit CUDABuffer(UserBuffer userBuffer, KernelArgument& kernelArgument);
    ~CUDABuffer();

    void resize(const size_t newBufferSize, const bool preserveData);

    void uploadData(const void* source, const size_t dataSize);
    void uploadData(CUstream stream, const void* source, const size_t dataSize, CUevent startEvent, CUevent endEvent);
    void uploadData(CUstream stream, const CUDABuffer* source, const size_t dataSize, CUevent startEvent, CUevent endEvent);

    void downloadData(void* destination, const size_t dataSize) const;
    void downloadData(CUstream stream, void* destination, const size_t dataSize, CUevent startEvent, CUevent endEvent) const;

    ArgumentId getKernelArgumentId() const;
    size_t getBufferSize() const;
    size_t getElementSize() const;
    ArgumentDataType getDataType() const;
    ArgumentMemoryLocation getMemoryLocation() const;
    ArgumentAccessType getAccessType() const;
    const CUdeviceptr* getBuffer() const;
    CUdeviceptr* getBuffer();

private:
    KernelArgument& kernelArgument;
    size_t bufferSize;
    CUdeviceptr buffer;
    void* hostBufferRaw;
    bool userOwned;
};

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
