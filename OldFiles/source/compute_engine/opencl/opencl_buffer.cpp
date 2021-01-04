#ifdef KTT_PLATFORM_OPENCL

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <compute_engine/opencl/opencl_buffer.h>
#include <compute_engine/opencl/opencl_event.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

OpenCLBuffer::OpenCLBuffer(const cl_context context, KernelArgument& kernelArgument) :
    context(context),
    kernelArgument(kernelArgument),
    bufferSize(kernelArgument.getDataSizeInBytes()),
    openclMemoryFlag(getOpenCLMemoryType(kernelArgument.getAccessType())),
    rawBuffer(nullptr),
    userOwned(false)
{
    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        openclMemoryFlag = openclMemoryFlag | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        rawBuffer = clSVMAlloc(context, openclMemoryFlag, bufferSize, 0);
            
        if (rawBuffer == nullptr)
        {
            throw std::runtime_error("Failed to allocate unified memory buffer");
        }

        #else
        throw std::runtime_error("Unified memory buffers are not supported on this platform");
        #endif
    }
    else
    {
        if (getMemoryLocation() == ArgumentMemoryLocation::Host)
        {
            openclMemoryFlag = openclMemoryFlag | CL_MEM_ALLOC_HOST_PTR;
        }
        else if (getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
        {
            openclMemoryFlag = openclMemoryFlag | CL_MEM_USE_HOST_PTR;
            rawBuffer = kernelArgument.getData();
        }

        cl_int result;
        buffer = clCreateBuffer(context, openclMemoryFlag, bufferSize, rawBuffer, &result);
        checkOpenCLError(result, "clCreateBuffer");
    }
}

OpenCLBuffer::OpenCLBuffer(UserBuffer userBuffer, KernelArgument& kernelArgument) :
    context(nullptr),
    kernelArgument(kernelArgument),
    bufferSize(kernelArgument.getDataSizeInBytes()),
    openclMemoryFlag(getOpenCLMemoryType(kernelArgument.getAccessType())),
    rawBuffer(nullptr),
    userOwned(true)
{
    if (userBuffer == nullptr)
    {
        throw std::runtime_error("The provided user OpenCL buffer is not valid");
    }

    buffer = static_cast<cl_mem>(userBuffer);
    checkOpenCLError(clGetMemObjectInfo(buffer, CL_MEM_CONTEXT, sizeof(cl_context), &context, nullptr), "clGetMemObjectInfo");

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        openclMemoryFlag = openclMemoryFlag | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        rawBuffer = userBuffer;
        #else
        throw std::runtime_error("Unified memory buffers are not supported on this platform");
        #endif
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        openclMemoryFlag = openclMemoryFlag | CL_MEM_ALLOC_HOST_PTR;
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        openclMemoryFlag = openclMemoryFlag | CL_MEM_USE_HOST_PTR;
        rawBuffer = kernelArgument.getData();
    }
}

OpenCLBuffer::~OpenCLBuffer()
{
    if (userOwned)
    {
        return;
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        clSVMFree(context, rawBuffer);
        #endif
    }
    else
    {
        checkOpenCLError(clReleaseMemObject(buffer), "clReleaseMemObject");
    }
}

void OpenCLBuffer::resize(cl_command_queue queue, const size_t newBufferSize, const bool preserveData)
{
    if (isZeroCopy())
    {
        throw std::runtime_error("Cannot resize buffer with CL_MEM_USE_HOST_PTR flag");
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        throw std::runtime_error("Unsupported SVM buffer operation");
    }

    if (bufferSize == newBufferSize)
    {
        return;
    }

    if (!preserveData)
    {
        checkOpenCLError(clReleaseMemObject(buffer), "clReleaseMemObject");
        cl_int result;
        buffer = clCreateBuffer(context, openclMemoryFlag, newBufferSize, rawBuffer, &result);
        checkOpenCLError(result, "clCreateBuffer");
    }
    else
    {
        cl_mem newBuffer;
        cl_int result;
        auto event = std::make_unique<OpenCLEvent>(0, true);

        newBuffer = clCreateBuffer(context, openclMemoryFlag, newBufferSize, rawBuffer, &result);
        checkOpenCLError(result, "clCreateBuffer");
        result = clEnqueueCopyBuffer(queue, buffer, newBuffer, 0, 0, std::min(bufferSize, newBufferSize), 0, nullptr, event->getEvent());
        checkOpenCLError(result, "clEnqueueCopyBuffer");

        event->setReleaseFlag();
        checkOpenCLError(clWaitForEvents(1, event->getEvent()), "clWaitForEvents");

        checkOpenCLError(clReleaseMemObject(buffer), "clReleaseMemObject");
        buffer = newBuffer;
    }

    bufferSize = newBufferSize;
}

void OpenCLBuffer::uploadData(cl_command_queue queue, const void* source, const size_t dataSize, cl_event* recordingEvent)
{
    if (bufferSize < dataSize)
    {
        resize(queue, dataSize, false);
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        std::memcpy(rawBuffer, source, dataSize);
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        if (recordingEvent == nullptr)
        {
            cl_int result = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, dataSize, source, 0, nullptr, nullptr);
            checkOpenCLError(result, "clEnqueueWriteBuffer");
        }
        else
        {
            cl_int result = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, dataSize, source, 0, nullptr, recordingEvent);
            checkOpenCLError(result, "clEnqueueWriteBuffer");
        }
    }
    else
    {
        // Asynchronous buffer operations on mapped memory are currently not supported
        cl_int result;
        void* destination = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize, 0, nullptr, nullptr, &result);
        checkOpenCLError(result, "clEnqueueMapBuffer");

        std::memcpy(destination, source, dataSize);
        checkOpenCLError(clEnqueueUnmapMemObject(queue, buffer, destination, 0, nullptr, recordingEvent), "clEnqueueUnmapMemObject");
    }
}
    
void OpenCLBuffer::uploadData(cl_command_queue queue, const cl_mem source, const size_t dataSize, cl_event* recordingEvent)
{
    if (bufferSize < dataSize)
    {
        resize(queue, dataSize, false);
    }

    if (recordingEvent == nullptr)
    {
        throw std::runtime_error("Recording event for buffer copying operation cannot be null");
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        throw std::runtime_error("Unsupported SVM buffer operation");
    }

    cl_int result = clEnqueueCopyBuffer(queue, source, buffer, 0, 0, dataSize, 0, nullptr, recordingEvent);
    checkOpenCLError(result, "clEnqueueCopyBuffer");
}

void OpenCLBuffer::downloadData(cl_command_queue queue, void* destination, const size_t dataSize, cl_event* recordingEvent) const
{
    if (bufferSize < dataSize)
    {
        throw std::runtime_error("Size of data to download is larger than size of buffer");
    }

    if (getMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        std::memcpy(destination, rawBuffer, dataSize);
    }
    else if (getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        if (recordingEvent == nullptr)
        {
            cl_int result = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, dataSize, destination, 0, nullptr, nullptr);
            checkOpenCLError(result, "clEnqueueReadBuffer");
        }
        else
        {
            cl_int result = clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, dataSize, destination, 0, nullptr, recordingEvent);
            checkOpenCLError(result, "clEnqueueReadBuffer");
        }
    }
    else
    {
        // Asynchronous buffer operations on mapped memory are currently not supported
        cl_int result;
        void* source = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, dataSize, 0, nullptr, nullptr, &result);
        checkOpenCLError(result, "clEnqueueMapBuffer");

        std::memcpy(destination, source, dataSize);
        checkOpenCLError(clEnqueueUnmapMemObject(queue, buffer, source, 0, nullptr, recordingEvent), "clEnqueueUnmapMemObject");
    }
}

cl_context OpenCLBuffer::getContext() const
{
    return context;
}

ArgumentId OpenCLBuffer::getKernelArgumentId() const
{
    return kernelArgument.getId();
}

size_t OpenCLBuffer::getBufferSize() const
{
    return bufferSize;
}

size_t OpenCLBuffer::getElementSize() const
{
    return kernelArgument.getElementSizeInBytes();
}

ArgumentDataType OpenCLBuffer::getDataType() const
{
    return kernelArgument.getDataType();
}

ArgumentMemoryLocation OpenCLBuffer::getMemoryLocation() const
{
    return kernelArgument.getMemoryLocation();
}

ArgumentAccessType OpenCLBuffer::getAccessType() const
{
    return kernelArgument.getAccessType();
}

cl_mem_flags OpenCLBuffer::getOpenclMemoryFlag() const
{
    return openclMemoryFlag;
}
    
cl_mem OpenCLBuffer::getBuffer() const
{
    return buffer;
}

void* OpenCLBuffer::getRawBuffer() const
{
    return rawBuffer;
}

bool OpenCLBuffer::isZeroCopy() const
{
    return static_cast<bool>(openclMemoryFlag & CL_MEM_USE_HOST_PTR);
}

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
