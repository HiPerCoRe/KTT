#ifdef KTT_PLATFORM_CUDA

#include <stdexcept>
#include <compute_engine/cuda/cuda_stream.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

CUDAStream::CUDAStream(const QueueId id) :
    id(id),
    owningStream(true)
{
    checkCUDAError(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
}

CUDAStream::CUDAStream(const QueueId id, UserQueue stream) :
    id(id),
    owningStream(false)
{
    this->stream = static_cast<CUstream>(stream);

    if (this->stream == nullptr)
    {
        throw std::runtime_error("The provided user CUDA stream is not valid");
    }
}

CUDAStream::~CUDAStream()
{
    if (owningStream)
    {
        checkCUDAError(cuStreamDestroy(stream), "cuStreamDestroy");
    }
}

CUstream CUDAStream::getStream() const
{
    return stream;
}

QueueId CUDAStream::getId() const
{
    return id;
}

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
