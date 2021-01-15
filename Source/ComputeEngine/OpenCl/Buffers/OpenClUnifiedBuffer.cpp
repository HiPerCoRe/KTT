#ifdef KTT_API_OPENCL

#include <cstring>

#include <ComputeEngine/OpenCl/Buffers/OpenClUnifiedBuffer.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Timer.h>

namespace ktt
{

OpenClUnifiedBuffer::OpenClUnifiedBuffer(const KernelArgument& argument, ActionIdGenerator& generator,
    const OpenClContext& context) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(context.GetContext()),
    m_SvmBuffer(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(false)
{
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");

#ifdef CL_VERSION_2_0
    m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    m_SvmBuffer = clSVMAlloc(m_Context, m_MemoryFlags, m_BufferSize, 0);
            
    if (m_SvmBuffer == nullptr)
    {
        throw KttException("Failed to allocate unified memory buffer");
    }

#else
    throw KttException("Unified memory buffers are not supported on this platform");
#endif
}

OpenClUnifiedBuffer::OpenClUnifiedBuffer(const KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(nullptr),
    m_SvmBuffer(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(true)
{
    KttAssert(argument.GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user OpenCL buffer is not valid");
    }

#ifdef CL_VERSION_2_0
    m_MemoryFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    m_SvmBuffer = userBuffer;
#else
    throw KttException("Unified memory buffers are not supported on this platform");
#endif
}

OpenClUnifiedBuffer::~OpenClUnifiedBuffer()
{
    if (m_UserOwned)
    {
        return;
    }

#ifdef CL_VERSION_2_0
    clSVMFree(m_Context, m_SvmBuffer);
#endif
}

std::unique_ptr<OpenClTransferAction> OpenClUnifiedBuffer::UploadData([[maybe_unused]] const OpenClCommandQueue& queue,
    const void* source, const size_t dataSize)
{
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

std::unique_ptr<OpenClTransferAction> OpenClUnifiedBuffer::CopyData([[maybe_unused]] const OpenClCommandQueue& queue,
    [[maybe_unused]] const OpenClBuffer& source, [[maybe_unused]] const size_t dataSize)
{
    throw KttException("Copy operation on unified OpenCL buffer is not supported");
}

void OpenClUnifiedBuffer::Resize([[maybe_unused]] const OpenClCommandQueue& queue, [[maybe_unused]] const size_t newSize,
    [[maybe_unused]] const bool preserveData)
{
    throw KttException("Resize operation on unified OpenCL buffer is not supported");
}

ArgumentId OpenClUnifiedBuffer::GetArgumentId() const
{
    return m_Argument.GetId();
}

ArgumentAccessType OpenClUnifiedBuffer::GetAccessType() const
{
    return m_Argument.GetAccessType();
}

ArgumentMemoryLocation OpenClUnifiedBuffer::GetMemoryLocation() const
{
    return m_Argument.GetMemoryLocation();
}

cl_mem OpenClUnifiedBuffer::GetBuffer() const
{
    throw KttException("Unified OpenCL buffer does not have cl_mem object");
}

void* OpenClUnifiedBuffer::GetRawBuffer()
{
    return m_SvmBuffer;
}

size_t OpenClUnifiedBuffer::GetSize() const
{
    return m_BufferSize;
}

} // namespace ktt

#endif // KTT_API_OPENCL
