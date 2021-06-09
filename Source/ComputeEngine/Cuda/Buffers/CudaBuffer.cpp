#ifdef KTT_API_CUDA

#include <ComputeEngine/Cuda/Buffers/CudaBuffer.h>

namespace ktt
{

CudaBuffer::CudaBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator) :
    m_Argument(argument),
    m_Generator(generator),
    m_BufferSize(argument.GetDataSize()),
    m_Buffer(0),
    m_UserOwned(false)
{}

CudaBuffer::CudaBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator,
    [[maybe_unused]] ComputeBuffer userBuffer) :
    m_Argument(argument),
    m_Generator(generator),
    m_BufferSize(argument.GetDataSize()),
    m_Buffer(0),
    m_UserOwned(true)
{}

const CUdeviceptr* CudaBuffer::GetBuffer() const
{
    return &m_Buffer;
}

CUdeviceptr* CudaBuffer::GetBuffer()
{
    return &m_Buffer;
}

ArgumentId CudaBuffer::GetArgumentId() const
{
    return m_Argument.GetId();
}

ArgumentAccessType CudaBuffer::GetAccessType() const
{
    return m_Argument.GetAccessType();
}

ArgumentMemoryLocation CudaBuffer::GetMemoryLocation() const
{
    return m_Argument.GetMemoryLocation();
}

size_t CudaBuffer::GetSize() const
{
    return m_BufferSize;
}

} // namespace ktt

#endif // KTT_API_CUDA
