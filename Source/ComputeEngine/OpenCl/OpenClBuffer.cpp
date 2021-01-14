#ifdef KTT_API_OPENCL

#include <algorithm>
#include <memory>

#include <ComputeEngine/OpenCl/OpenClBuffer.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

OpenClBuffer::OpenClBuffer(KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(context.GetContext()),
    m_RawBuffer(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(false)
{
    if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
        m_RawBuffer = clSVMAlloc(m_Context, m_MemoryFlags, m_BufferSize, 0);
            
        if (m_RawBuffer == nullptr)
        {
            throw KttException("Failed to allocate unified memory buffer");
        }

        #else
        throw KttException("Unified memory buffers are not supported on this platform");
        #endif
    }
    else
    {
        if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Host)
        {
            m_MemoryFlags |= CL_MEM_ALLOC_HOST_PTR;
        }
        else if (argument.GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
        {
            m_MemoryFlags |= CL_MEM_USE_HOST_PTR;
            m_RawBuffer = argument.GetData();
        }

        cl_int result;
        m_Buffer = clCreateBuffer(m_Context, m_MemoryFlags, m_BufferSize, m_RawBuffer, &result);
        CheckError(result, "clCreateBuffer");
    }
}

OpenClBuffer::OpenClBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(nullptr),
    m_RawBuffer(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(true)
{
    if (userBuffer == nullptr)
    {
        throw KttException("The provided user OpenCL buffer is not valid");
    }

    m_Buffer = static_cast<cl_mem>(userBuffer);
    CheckError(clGetMemObjectInfo(m_Buffer, CL_MEM_CONTEXT, sizeof(m_Context), &m_Context, nullptr), "clGetMemObjectInfo");

    if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
        m_RawBuffer = userBuffer;
        #else
        throw KttException("Unified memory buffers are not supported on this platform");
        #endif
    }
    else if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        m_MemoryFlags |= CL_MEM_ALLOC_HOST_PTR;
    }
    else if (argument.GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        m_MemoryFlags |= CL_MEM_USE_HOST_PTR;
        m_RawBuffer = argument.GetData();
    }
}

OpenClBuffer::~OpenClBuffer()
{
    if (m_UserOwned)
    {
        return;
    }

    if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        #ifdef CL_VERSION_2_0
        clSVMFree(m_Context, m_RawBuffer);
        #endif
    }
    else
    {
        CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
    }
}

std::unique_ptr<OpenClTransferAction> OpenClBuffer::UploadData(const OpenClCommandQueue& queue, const void* source,
    const size_t dataSize)
{
    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id);

    if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        std::memcpy(m_RawBuffer, source, dataSize);
    }
    else if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        cl_int result = clEnqueueWriteBuffer(queue.GetQueue(), m_Buffer, CL_FALSE, 0, dataSize, source, 0, nullptr,
            action->GetEvent().GetEvent());
        CheckError(result, "clEnqueueWriteBuffer");
    }
    else
    {
        // Asynchronous buffer operations on mapped memory are currently not supported
        cl_int result;
        void* destination = clEnqueueMapBuffer(queue.GetQueue(), m_Buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize, 0, nullptr,
            nullptr, &result);
        CheckError(result, "clEnqueueMapBuffer");

        std::memcpy(destination, source, dataSize);
        CheckError(clEnqueueUnmapMemObject(queue.GetQueue(), m_Buffer, destination, 0, nullptr, action->GetEvent().GetEvent()),
            "clEnqueueUnmapMemObject");
    }

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClBuffer::DownloadData(const OpenClCommandQueue& queue, void* destination,
    const size_t dataSize) const
{
    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id);

    if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        std::memcpy(destination, m_RawBuffer, dataSize);
    }
    else if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        cl_int result = clEnqueueReadBuffer(queue.GetQueue(), m_Buffer, CL_FALSE, 0, dataSize, destination, 0, nullptr,
            action->GetEvent().GetEvent());
        CheckError(result, "clEnqueueReadBuffer");
    }
    else
    {
        // Asynchronous buffer operations on mapped memory are currently not supported
        cl_int result;
        void* source = clEnqueueMapBuffer(queue.GetQueue(), m_Buffer, CL_TRUE, CL_MAP_READ, 0, dataSize, 0, nullptr,
            nullptr, &result);
        CheckError(result, "clEnqueueMapBuffer");

        std::memcpy(destination, source, dataSize);
        CheckError(clEnqueueUnmapMemObject(queue.GetQueue(), m_Buffer, source, 0, nullptr, action->GetEvent().GetEvent()),
            "clEnqueueUnmapMemObject");
    }

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClBuffer::CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
    const size_t dataSize)
{
    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id);

    if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        throw KttException("Unsupported SVM buffer operation");
    }

    cl_int result = clEnqueueCopyBuffer(queue.GetQueue(), source.GetBuffer(), m_Buffer, 0, 0, dataSize, 0, nullptr,
        action->GetEvent().GetEvent());
    CheckError(result, "clEnqueueCopyBuffer");

    return action;
}

void OpenClBuffer::Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData)
{
    if (IsZeroCopy())
    {
        throw KttException("Cannot resize buffer with CL_MEM_USE_HOST_PTR flag");
    }

    if (m_Argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified)
    {
        throw KttException("Unsupported SVM buffer operation");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    if (!preserveData)
    {
        CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
        cl_int result;
        m_Buffer = clCreateBuffer(m_Context, m_MemoryFlags, newSize, m_RawBuffer, &result);
        CheckError(result, "clCreateBuffer");
    }
    else
    {
        cl_mem newBuffer;
        cl_int result;
        auto event = std::make_unique<OpenClEvent>();

        newBuffer = clCreateBuffer(m_Context, m_MemoryFlags, newSize, m_RawBuffer, &result);
        CheckError(result, "clCreateBuffer");
        result = clEnqueueCopyBuffer(queue.GetQueue(), m_Buffer, newBuffer, 0, 0, std::min(m_BufferSize, newSize), 0,
            nullptr, event->GetEvent());
        CheckError(result, "clEnqueueCopyBuffer");

        event->SetReleaseFlag();
        event->WaitForFinish();

        CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
        m_Buffer = newBuffer;
    }

    m_BufferSize = newSize;
}

KernelArgument& OpenClBuffer::GetArgument() const
{
    return m_Argument;
}

cl_context OpenClBuffer::GetContext() const
{
    return m_Context;
}

cl_mem OpenClBuffer::GetBuffer() const
{
    return m_Buffer;
}

void* OpenClBuffer::GetRawBuffer() const
{
    return m_RawBuffer;
}

size_t OpenClBuffer::GetSize() const
{
    return m_BufferSize;
}

bool OpenClBuffer::IsZeroCopy() const
{
    return static_cast<bool>(m_MemoryFlags & CL_MEM_USE_HOST_PTR);
}

cl_mem_flags OpenClBuffer::GetMemoryFlags()
{
    switch (m_Argument.GetAccessType())
    {
    case ArgumentAccessType::ReadWrite:
        return CL_MEM_READ_WRITE;
    case ArgumentAccessType::Read:
        return CL_MEM_READ_ONLY;
    case ArgumentAccessType::Write:
        return CL_MEM_WRITE_ONLY;
    default:
        KttError("Unhandled argument access type value");
        return 0;
    }
}

} // namespace ktt

#endif // KTT_API_OPENCL
