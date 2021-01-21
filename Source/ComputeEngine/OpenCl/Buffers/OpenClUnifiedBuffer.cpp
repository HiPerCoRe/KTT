#ifdef KTT_API_OPENCL
#ifdef CL_VERSION_2_0

#include <cstring>
#include <string>

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClUnifiedBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer.h>

namespace ktt
{

OpenClUnifiedBuffer::OpenClUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context) :
    OpenClBuffer(argument, generator, context),
    m_SvmBuffer(nullptr)
{
    Logger::LogDebug("Initializing OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");

    m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    m_SvmBuffer = clSVMAlloc(m_Context, m_MemoryFlags, m_BufferSize, 0);
            
    if (m_SvmBuffer == nullptr)
    {
        throw KttException("Failed to allocate unified memory buffer");
    }
}

OpenClUnifiedBuffer::OpenClUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer) :
    OpenClBuffer(argument, generator),
    m_SvmBuffer(nullptr)
{
    Logger::LogDebug("Initializing OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user OpenCL buffer is not valid");
    }

    m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    m_SvmBuffer = userBuffer;
}

OpenClUnifiedBuffer::~OpenClUnifiedBuffer()
{
    Logger::LogDebug("Releasing OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        return;
    }

    clSVMFree(m_Context, m_SvmBuffer);
}

std::unique_ptr<OpenClTransferAction> OpenClUnifiedBuffer::UploadData([[maybe_unused]] const OpenClCommandQueue& queue,
    const void* source, const size_t dataSize)
{
    Logger::LogDebug("Uploading data into OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id, false);

    Timer timer;
    timer.Start();

    std::memcpy(m_SvmBuffer, source, dataSize);

    timer.Stop();
    action->SetDuration(timer.GetElapsedTime());
    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClUnifiedBuffer::DownloadData([[maybe_unused]] const OpenClCommandQueue& queue,
    void* destination, const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<OpenClTransferAction>(id, false);

    Timer timer;
    timer.Start();

    std::memcpy(destination, m_SvmBuffer, dataSize);

    timer.Stop();
    action->SetDuration(timer.GetElapsedTime());
    return action;
}

std::unique_ptr<OpenClTransferAction> OpenClUnifiedBuffer::CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into OpenCL unified buffer with id " + std::to_string(m_Argument.GetId())
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
    auto action = std::make_unique<OpenClTransferAction>(id, false);

    Timer timer;
    timer.Start();

    std::vector<uint8_t> data(dataSize);
    auto action = source.DownloadData(queue, data.data(), dataSize);
    action->WaitForFinish();
    std::memcpy(m_SvmBuffer, data.data(), dataSize);

    timer.Stop();
    action->SetDuration(timer.GetElapsedTime());
    return action;
}

void OpenClUnifiedBuffer::Resize([[maybe_unused]] const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing OpenCL unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        throw KttException("Resize operation on user owned buffer is not supported");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    void* newSvmBuffer = clSVMAlloc(m_Context, m_MemoryFlags, newSize, 0);

    if (m_SvmBuffer == nullptr)
    {
        throw KttException("Failed to allocate unified memory buffer");
    }

    if (preserveData)
    {
        std::memcpy(newSvmBuffer, m_SvmBuffer, std::min(m_BufferSize, newSize));
    }

    clSVMFree(m_Context, m_SvmBuffer);
    m_SvmBuffer = newSvmBuffer;
    m_BufferSize = newSize;
}

cl_mem OpenClUnifiedBuffer::GetBuffer() const
{
    throw KttException("Unified OpenCL buffer does not have cl_mem object");
}

void* OpenClUnifiedBuffer::GetRawBuffer()
{
    return m_SvmBuffer;
}

} // namespace ktt

#endif // CL_VERSION_2_0
#endif // KTT_API_OPENCL
