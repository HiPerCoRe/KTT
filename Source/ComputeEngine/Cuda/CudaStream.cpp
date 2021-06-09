#ifdef KTT_API_CUDA

#include <string>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaStream::CudaStream(const QueueId id) :
    m_Id(id),
    m_OwningStream(true)
{
    Logger::LogDebug("Initializing CUDA stream with id " + std::to_string(id));
    CheckError(cuStreamCreate(&m_Stream, CU_STREAM_DEFAULT), "cuStreamCreate");
}

CudaStream::CudaStream(const QueueId id, ComputeQueue stream) :
    m_Id(id),
    m_OwningStream(false)
{
    Logger::LogDebug("Initializing CUDA stream with id " + std::to_string(id));
    m_Stream = static_cast<CUstream>(stream);

    if (m_Stream == nullptr)
    {
        throw KttException("The provided user CUDA stream is not valid");
    }
}

CudaStream::~CudaStream()
{
    Logger::LogDebug("Releasing CUDA stream with id " + std::to_string(m_Id));

    if (m_OwningStream)
    {
        CheckError(cuStreamDestroy(m_Stream), "cuStreamDestroy");
    }
}

void CudaStream::Synchronize() const
{
    Logger::LogDebug("Synchronizing CUDA stream with id " + std::to_string(m_Id));
    CheckError(cuStreamSynchronize(m_Stream), "cuStreamSynchronize");
}

CUstream CudaStream::GetStream() const
{
    return m_Stream;
}

QueueId CudaStream::GetId() const
{
    return m_Id;
}

} // namespace ktt

#endif // KTT_API_CUDA
