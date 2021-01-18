#ifdef KTT_API_OPENCL

#include <algorithm>
#include <string>

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClDeviceBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClDeviceBuffer::OpenClDeviceBuffer(const KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(context.GetContext()),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(false)
{
    Logger::LogDebug("Initializing OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Device, "Argument memory location mismatch");

    cl_int result;
    m_Buffer = clCreateBuffer(m_Context, m_MemoryFlags, m_BufferSize, nullptr, &result);
    CheckError(result, "clCreateBuffer");
}

OpenClDeviceBuffer::OpenClDeviceBuffer(const KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(true)
{
    Logger::LogDebug("Initializing OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Device, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user OpenCL buffer is not valid");
    }

    m_Buffer = static_cast<cl_mem>(userBuffer);
    CheckError(clGetMemObjectInfo(m_Buffer, CL_MEM_CONTEXT, sizeof(m_Context), &m_Context, nullptr), "clGetMemObjectInfo");
}

OpenClDeviceBuffer::~OpenClDeviceBuffer()
{
    Logger::LogDebug("Releasing OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        return;
    }

    CheckError(clReleaseMemObject(m_Buffer), "clReleaseMemObject");
}

std::unique_ptr<OpenClTransferAction> OpenClDeviceBuffer::UploadData(const OpenClCommandQueue& queue, const void* source,
    const size_t dataSize)
{
    Logger::LogDebug("Uploading data into OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id, true);

    cl_int result = clEnqueueWriteBuffer(queue.GetQueue(), m_Buffer, CL_FALSE, 0, dataSize, source, 0, nullptr,
        action->GetEvent());
    CheckError(result, "clEnqueueWriteBuffer");
    action->SetReleaseFlag();

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClDeviceBuffer::DownloadData(const OpenClCommandQueue& queue, void* destination,
    const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id, true);

    cl_int result = clEnqueueReadBuffer(queue.GetQueue(), m_Buffer, CL_FALSE, 0, dataSize, destination, 0, nullptr,
        action->GetEvent());
    CheckError(result, "clEnqueueReadBuffer");
    action->SetReleaseFlag();

    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClDeviceBuffer::CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into OpenCL device buffer with id " + std::to_string(m_Argument.GetId())
        + " from buffer with id " + std::to_string(source.GetArgumentId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of target buffer");
    }

    if (source.GetSize() < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of source buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id, true);

    cl_int result = clEnqueueCopyBuffer(queue.GetQueue(), source.GetBuffer(), m_Buffer, 0, 0, dataSize, 0, nullptr,
        action->GetEvent());
    CheckError(result, "clEnqueueCopyBuffer");
    action->SetReleaseFlag();

    return action;
}

void OpenClDeviceBuffer::Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing OpenCL device buffer with id " + std::to_string(m_Argument.GetId()));

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

ArgumentId OpenClDeviceBuffer::GetArgumentId() const
{
    return m_Argument.GetId();
}

ArgumentAccessType OpenClDeviceBuffer::GetAccessType() const
{
    return m_Argument.GetAccessType();
}

ArgumentMemoryLocation OpenClDeviceBuffer::GetMemoryLocation() const
{
    return m_Argument.GetMemoryLocation();
}

cl_mem OpenClDeviceBuffer::GetBuffer() const
{
    return m_Buffer;
}

void* OpenClDeviceBuffer::GetRawBuffer()
{
    return static_cast<void*>(&m_Buffer);
}

size_t OpenClDeviceBuffer::GetSize() const
{
    return m_BufferSize;
}

} // namespace ktt

#endif // KTT_API_OPENCL
