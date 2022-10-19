#ifdef KTT_API_CUDA

#include <algorithm>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Actions/CudaTransferAction.h>
#include <ComputeEngine/Cuda/Buffers/CudaDeviceBuffer.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaDeviceBuffer::CudaDeviceBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator) :
    CudaBuffer(argument, generator)
{
    Logger::LogDebug("Initializing CUDA device buffer with id " + m_Argument.GetId());
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Device, "Argument memory location mismatch");
    CheckError(cuMemAlloc(&m_Buffer, m_BufferSize), "cuMemAlloc");
}

CudaDeviceBuffer::CudaDeviceBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator, ComputeBuffer userBuffer) :
    CudaBuffer(argument, generator, userBuffer)
{
    Logger::LogDebug("Initializing CUDA device buffer with id " + m_Argument.GetId());
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Device, "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user CUDA buffer is not valid");
    }

    m_Buffer = reinterpret_cast<CUdeviceptr>(userBuffer);
}

CudaDeviceBuffer::~CudaDeviceBuffer()
{
    Logger::LogDebug("Releasing CUDA device buffer with id " + m_Argument.GetId());

    if (m_UserOwned)
    {
        return;
    }

    CheckError(cuMemFree(m_Buffer), "cuMemFree");
}

std::unique_ptr<CudaTransferAction> CudaDeviceBuffer::UploadData(const CudaStream& stream, const void* source,
    const size_t dataSize)
{
    Logger::LogDebug("Uploading data into CUDA device buffer with id " + m_Argument.GetId());

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<CudaTransferAction>(id, stream.GetId());

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyHtoDAsync(m_Buffer, source, dataSize, stream.GetStream()), "cuMemcpyHtoDAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

std::unique_ptr<CudaTransferAction> CudaDeviceBuffer::DownloadData(const CudaStream& stream, void* destination,
    const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from CUDA device buffer with id " + m_Argument.GetId());

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<CudaTransferAction>(id, stream.GetId());

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyDtoHAsync(destination, m_Buffer, dataSize, stream.GetStream()), "cuMemcpyDtoHAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

std::unique_ptr<CudaTransferAction> CudaDeviceBuffer::CopyData(const CudaStream& stream, const CudaBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into CUDA device buffer with id " + m_Argument.GetId() + " from buffer with id "
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
    auto action = std::make_unique<CudaTransferAction>(id, stream.GetId());

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuMemcpyDtoDAsync(m_Buffer, *source.GetBuffer(), dataSize, stream.GetStream()), "cuMemcpyDtoDAsync");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
}

void CudaDeviceBuffer::Resize(const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing CUDA device buffer with id " + m_Argument.GetId());

    if (m_UserOwned)
    {
        throw KttException("Resize operation on user owned buffer is not supported");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    CUdeviceptr newBuffer;
    CheckError(cuMemAlloc(&newBuffer, newSize), "cuMemAlloc");

    if (preserveData)
    {
        CheckError(cuMemcpyDtoD(newBuffer, m_Buffer, std::min(m_BufferSize, newSize)), "cuMemcpyDtoD");
    }

    CheckError(cuMemFree(m_Buffer), "cuMemFree");
    m_Buffer = newBuffer;
    m_BufferSize = newSize;
}

} // namespace ktt

#endif // KTT_API_CUDA
