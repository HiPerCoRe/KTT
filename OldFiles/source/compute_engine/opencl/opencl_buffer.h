#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <CL/cl.h>
#include <compute_engine/opencl/opencl_event.h>
#include <kernel_argument/kernel_argument.h>
#include <ktt_types.h>

namespace ktt
{

class OpenCLBuffer
{
public:
    explicit OpenCLBuffer(const cl_context context, KernelArgument& kernelArgument);
    explicit OpenCLBuffer(UserBuffer userBuffer, KernelArgument& kernelArgument);
    ~OpenCLBuffer();

    void resize(cl_command_queue queue, const size_t newBufferSize, const bool preserveData);
    void uploadData(cl_command_queue queue, const void* source, const size_t dataSize, cl_event* recordingEvent);
    void uploadData(cl_command_queue queue, const cl_mem source, const size_t dataSize, cl_event* recordingEvent);
    void downloadData(cl_command_queue queue, void* destination, const size_t dataSize, cl_event* recordingEvent) const;

    cl_context getContext() const;
    ArgumentId getKernelArgumentId() const;
    size_t getBufferSize() const;
    size_t getElementSize() const;
    ArgumentDataType getDataType() const;
    ArgumentMemoryLocation getMemoryLocation() const;
    ArgumentAccessType getAccessType() const;
    cl_mem_flags getOpenclMemoryFlag() const;
    cl_mem getBuffer() const;
    void* getRawBuffer() const;
    bool isZeroCopy() const;

private:
    cl_context context;
    KernelArgument& kernelArgument;
    size_t bufferSize;
    cl_mem_flags openclMemoryFlag;
    cl_mem buffer;
    void* rawBuffer;
    bool userOwned;
};

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
