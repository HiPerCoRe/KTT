#ifdef KTT_API_CUDA

#include <algorithm>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Actions/CudaTransferAction.h>
#include <ComputeEngine/Cuda/Buffers/CudaHostBuffer.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaHostBuffer::CudaHostBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator) :
    CudaBuffer(argument, generator),
    m_RawBuffer(nullptr)
{
    Logger::LogDebug("Initializing CUDA host buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Host || GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy,
        "Argument memory location mismatch");

    if (GetMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        CheckError(cuMemAllocHost(&m_RawBuffer, m_BufferSize), "cuMemAllocHost");
    }
    else
    {
        m_RawBuffer = argument.GetData();
        CheckError(cuMemHostRegister(m_RawBuffer, m_BufferSize, CU_MEMHOSTREGISTER_DEVICEMAP), "cuMemHostRegister");
    }

    CheckError(cuMemHostGetDevicePointer(&m_Buffer, m_RawBuffer, 0), "cuMemHostGetDevicePointer");
}

CudaHostBuffer::CudaHostBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator, ComputeBuffer userBuffer) :
    CudaBuffer(argument, generator, userBuffer),
    m_RawBuffer(nullptr)
{
    Logger::LogDebug("Initializing CUDA host buffer with id " + std::to_string(m_Argument.GetId()));
    KttAssert(GetMemoryLocation() == ArgumentMemoryLocation::Host || GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy,
        "Argument memory location mismatch");

    if (userBuffer == nullptr)
    {
        throw KttException("The provided user CUDA buffer is not valid");
    }

    m_Buffer = reinterpret_cast<CUdeviceptr>(userBuffer);
}

CudaHostBuffer::~CudaHostBuffer()
{
    Logger::LogDebug("Releasing CUDA host buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        return;
    }

    if (GetMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        CheckError(cuMemFreeHost(m_RawBuffer), "cuMemFreeHost");
    }
    else
    {
        CheckError(cuMemHostUnregister(m_RawBuffer), "cuMemHostUnregister");
    }
}

std::unique_ptr<CudaTransferAction> CudaHostBuffer::UploadData(const CudaStream& stream, const void* source,
    const size_t dataSize)
{
    Logger::LogDebug("Uploading data into CUDA host buffer with id " + std::to_string(m_Argument.GetId()));

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

std::unique_ptr<CudaTransferAction> CudaHostBuffer::DownloadData(const CudaStream& stream, void* destination,
    const size_t dataSize) const
{
    Logger::LogDebug("Downloading data from CUDA host buffer with id " + std::to_string(m_Argument.GetId()));

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

std::unique_ptr<CudaTransferAction> CudaHostBuffer::CopyData(const CudaStream& stream, const CudaBuffer& source,
    const size_t dataSize)
{
    Logger::LogDebug("Copying data into CUDA host buffer with id " + std::to_string(m_Argument.GetId())
        + " from buffer with id " + std::to_string(source.GetArgumentId()));

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

void CudaHostBuffer::Resize(const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing CUDA host buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_UserOwned)
    {
        throw KttException("Resize operation on user owned buffer is not supported");
    }

    if (GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        throw KttException("Resize operation on registered host buffer is not supported");
    }

    if (m_BufferSize == newSize)
    {
        return;
    }

    void* newRawBuffer = nullptr;
    CUdeviceptr newBuffer;
    CheckError(cuMemAllocHost(&newRawBuffer, newSize), "cuMemAllocHost");
    CheckError(cuMemHostGetDevicePointer(&newBuffer, newRawBuffer, 0), "cuMemHostGetDevicePointer");

    if (preserveData)
    {
        CheckError(cuMemcpyDtoD(newBuffer, m_Buffer, std::min(m_BufferSize, newSize)), "cuMemcpyDtoD");
    }

    CheckError(cuMemFreeHost(m_RawBuffer), "cuMemFreeHost");
    m_RawBuffer = newRawBuffer;
    m_Buffer = newBuffer;
    m_BufferSize = newSize;
}

} // namespace ktt

#endif // KTT_API_CUDA
