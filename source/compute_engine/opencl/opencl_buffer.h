#pragma once

#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_command_queue.h"
#include "opencl_utility.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_type.h"

namespace ktt
{

class OpenclBuffer
{
public:
    explicit OpenclBuffer(const cl_context context, const size_t kernelArgumentId, const size_t bufferSize, const size_t elementSize,
        const ArgumentDataType& dataType, const ArgumentMemoryType& memoryType) :
        context(context),
        kernelArgumentId(kernelArgumentId),
        bufferSize(bufferSize),
        elementSize(elementSize),
        dataType(dataType),
        memoryType(memoryType),
        openclMemoryFlag(getOpenclMemoryType(memoryType))
    {
        cl_int result;
        buffer = clCreateBuffer(context, openclMemoryFlag, bufferSize, nullptr, &result);
        checkOpenclError(result, "clCreateBuffer");
    }

    ~OpenclBuffer()
    {
        checkOpenclError(clReleaseMemObject(buffer), "clReleaseMemObject");
    }

    void uploadData(OpenclCommandQueue& queue, const void* source, const size_t dataSize)
    {
        if (bufferSize != dataSize)
        {
            checkOpenclError(clReleaseMemObject(buffer), "clReleaseMemObject");

            cl_int result;
            buffer = clCreateBuffer(context, openclMemoryFlag, dataSize, nullptr, &result);
            checkOpenclError(result, "clCreateBuffer");

            bufferSize = dataSize;
        }

        cl_int result = clEnqueueWriteBuffer(queue.getQueue(), buffer, CL_TRUE, 0, dataSize, source, 0, nullptr, nullptr);
        checkOpenclError(result, "clEnqueueWriteBuffer");
    }

    void downloadData(OpenclCommandQueue& queue, void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        cl_int result = clEnqueueReadBuffer(queue.getQueue(), buffer, CL_TRUE, 0, dataSize, destination, 0, nullptr, nullptr);
        checkOpenclError(result, "clEnqueueReadBuffer");
    }

    cl_context getContext() const
    {
        return context;
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

    cl_mem getBuffer() const
    {
        return buffer;
    }

    cl_mem_flags getOpenclMemoryFlag() const
    {
        return openclMemoryFlag;
    }

private:
    cl_context context;
    size_t kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryType memoryType;
    cl_mem buffer;
    cl_mem_flags openclMemoryFlag;
};

} // namespace ktt
