#ifdef KTT_API_CUDA

#include <algorithm>

#include <ComputeEngine/Cuda/Actions/CudaTransferAction.h>
#include <ComputeEngine/Cuda/Buffers/CudaUnifiedBuffer.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaUnifiedBuffer::CudaUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator) :
    CudaBuffer(argument, generator)
{
    Logger::LogDebug("Initializing CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");
    CheckError(cuMemAllocManaged(&m_Buffer, m_BufferSize, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
}

CudaUnifiedBuffer::CudaUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer) :
    CudaBuffer(argument, generator, userBuffer)
{
    Logger::LogDebug("Initializing CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Unified, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user CUDA buffer is not valid");
    }

    m_Buffer = reinterpret_cast<CUdeviceptr>(userBuffer);
}

CudaUnifiedBuffer::~CudaUnifiedBuffer()
{
    Logger::LogDebug("Releasing CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        return;
    }

    CheckError(cuMemFree(m_Buffer), "cuMemFree");
}

std::unique_ptr<CudaTransferAction> CudaUnifiedBuffer::UploadData(const CudaStream& stream, const void* source,
    const size_t dataSize)
{
    Logger::LogDebug("Uploading data into CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<CudaTransferAction>(id);

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyAsync(m_Buffer, reinterpret_cast<CUdeviceptr>(source), dataSize, stream.GetStream()), "cuMemcpyAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

std::unique_ptr<CudaTransferAction> CudaUnifiedBuffer::DownloadData(const CudaStream& stream, void* destination,
    const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateTransferId();
    auto action = std::make_unique<CudaTransferAction>(id);

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(destination), m_Buffer, dataSize, stream.GetStream()), "cuMemcpyAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

std::unique_ptr<CudaTransferAction> CudaUnifiedBuffer::CopyData(const CudaStream& stream, const CudaBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into CUDA unified buffer with id " + std::to_string(m_Argument.GetId())
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
    auto action = std::make_unique<CudaTransferAction>(id);

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyAsync(m_Buffer, *source.GetBuffer(), dataSize, stream.GetStream()), "cuMemcpyAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

void CudaUnifiedBuffer::Resize(const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing CUDA unified buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        throw KttException("Resize operation on user owned buffer is not supported");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    CUdeviceptr newBuffer;
    CheckError(cuMemAllocManaged(&newBuffer, newSize, CU_MEM_ATTACH_GLOBAL), "cuMemAlloc");

    if (preserveData)
    {
        CheckError(cuMemcpy(newBuffer, m_Buffer, std::min(m_BufferSize, newSize)), "cuMemcpy");
    }

    CheckError(cuMemFree(m_Buffer), "cuMemFree");
    m_Buffer = newBuffer;
    m_BufferSize = newSize;
}

} // namespace ktt

#endif // KTT_API_CUDA
