#ifdef KTT_API_OPENCL

#include <algorithm>
#include <cstring>
#include <string>

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClHostBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

OpenClHostBuffer::OpenClHostBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator,
    const OpenClContext& context) :
    OpenClBuffer(argument, generator, context),
    m_RawBuffer(nullptr)
{
    Logger::LogDebug("Initializing OpenCL host buffer with id " + m_Argument.GetId());
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Host
        || argument.GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy, "Argument memory location mismatch");

    if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        m_MemoryFlags |= CL_MEM_ALLOC_HOST_PTR;
    }
    else
    {
        m_MemoryFlags |= CL_MEM_USE_HOST_PTR;
        m_RawBuffer = argument.GetData();
    }

    cl_int result;
    m_Buffer = clCreateBuffer(m_Context, m_MemoryFlags, m_BufferSize, m_RawBuffer, &result);
    CheckError(result, "clCreateBuffer");
}

OpenClHostBuffer::OpenClHostBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator,
    ComputeBuffer userBuffer) :
    OpenClBuffer(argument, generator),
    m_RawBuffer(nullptr)
{
    Logger::LogDebug("Initializing OpenCL host buffer with id " + m_Argument.GetId());
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Host
        || argument.GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user OpenCL buffer is not valid");
    }

    m_Buffer = static_cast<cl_mem>(userBuffer);
    CheckError(clGetMemObjectInfo(m_Buffer, CL_MEM_CONTEXT, sizeof(m_Context), &m_Context, nullptr), "clGetMemObjectInfo");

    if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        m_MemoryFlags |= CL_MEM_ALLOC_HOST_PTR;
    }
    else
    {
        m_MemoryFlags |= CL_MEM_USE_HOST_PTR;
        m_RawBuffer = argument.GetData();
    }
}

OpenClHostBuffer::~OpenClHostBuffer()
{
    Logger::LogDebug("Releasing OpenCL host buffer with id " + m_Argument.GetId());

    if (m_UserOwned)
    {
        return;
    }

    CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
}

std::unique_ptr<OpenClTransferAction> OpenClHostBuffer::UploadData(const OpenClCommandQueue& queue, const void* source,
    const size_t dataSize)
{
    Logger::LogDebug("Uploading data into OpenCL host buffer with id " + m_Argument.GetId());

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<OpenClTransferAction>(id, queue.GetId(), false);

    Timer timer;
    timer.Start();

    cl_int result;
    void* destination = clEnqueueMapBuffer(queue.GetQueue(), m_Buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize, 0, nullptr,
        nullptr, &result);
    CheckError(result, "clEnqueueMapBuffer");

    std::memcpy(destination, source, dataSize);

    auto event = std::make_unique<OpenClEvent>();
    CheckError(clEnqueueUnmapMemObject(queue.GetQueue(), m_Buffer, destination, 0, nullptr, event->GetEvent()),
        "clEnqueueUnmapMemObject");
    event->SetReleaseFlag();
    event->WaitForFinish();

    timer.Stop();
    action->SetDuration(timer.GetElapsedTime());

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClHostBuffer::DownloadData(const OpenClCommandQueue& queue, void* destination,
    const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from OpenCL host buffer with id " + m_Argument.GetId());

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<OpenClTransferAction>(id, queue.GetId(), false);

    Timer timer;
    timer.Start();

    cl_int result;
    void* source = clEnqueueMapBuffer(queue.GetQueue(), m_Buffer, CL_TRUE, CL_MAP_READ, 0, dataSize, 0, nullptr,
        nullptr, &result);
    CheckError(result, "clEnqueueMapBuffer");

    std::memcpy(destination, source, dataSize);

    auto event = std::make_unique<OpenClEvent>();
    CheckError(clEnqueueUnmapMemObject(queue.GetQueue(), m_Buffer, source, 0, nullptr, event->GetEvent()),
        "clEnqueueUnmapMemObject");
    event->SetReleaseFlag();
    event->WaitForFinish();

    timer.Stop();
    action->SetDuration(timer.GetElapsedTime());

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClHostBuffer::CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into OpenCL host buffer with id " + m_Argument.GetId() + " from buffer with id "
        + source.GetArgumentId());

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of target buffer");
    }

    if (source.GetSize() < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of source buffer");
    }

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<OpenClTransferAction>(id, queue.GetId(), true);

    cl_int result = clEnqueueCopyBuffer(queue.GetQueue(), source.GetBuffer(), m_Buffer, 0, 0, dataSize, 0, nullptr,
        action->GetEvent());
    CheckError(result, "clEnqueueCopyBuffer");
    action->SetReleaseFlag();

    return action;
}

void OpenClHostBuffer::Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing OpenCL host buffer with id " + m_Argument.GetId());

    if (m_UserOwned)
    {
        throw KttException("Resize operation on user owned buffer is not supported");
    }

    if (IsZeroCopy())
    {
        throw KttException("Resize operation on buffer with CL_MEM_USE_HOST_PTR flag is not supported");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    cl_int result;
    cl_mem newBuffer = clCreateBuffer(m_Context, m_MemoryFlags, newSize, nullptr, &result);
    CheckError(result, "clCreateBuffer");

    if (preserveData)
    {
        auto event = std::make_unique<OpenClEvent>();
        result = clEnqueueCopyBuffer(queue.GetQueue(), m_Buffer, newBuffer, 0, 0, std::min(m_BufferSize, newSize), 0,
            nullptr, event->GetEvent());
        CheckError(result, "clEnqueueCopyBuffer");

        event->SetReleaseFlag();
        event->WaitForFinish();
    }

    CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
    m_Buffer = newBuffer;
    m_BufferSize = newSize;
}

cl_mem OpenClHostBuffer::GetBuffer() const
{
    return m_Buffer;
}

void* OpenClHostBuffer::GetRawBuffer()
{
    return static_cast<void*>(&m_Buffer);
}

bool OpenClHostBuffer::IsZeroCopy() const
{
    return static_cast<bool>(m_MemoryFlags & CL_MEM_USE_HOST_PTR);
}

} // namespace ktt

#endif // KTT_API_OPENCL
