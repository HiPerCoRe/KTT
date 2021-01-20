#pragma once

#ifdef KTT_API_CUDA

#include <cuda.h>

#include <KttTypes.h>

namespace ktt
{

class CudaStream
{
public:
    explicit CudaStream(const QueueId id);
    explicit CudaStream(const QueueId id, ComputeQueue stream);
    ~CudaStream();

    void Synchronize() const;

    CUstream GetStream() const;
    QueueId GetId() const;

private:
    CUstream m_Stream;
    QueueId m_Id;
    bool m_OwningStream;
};

} // namespace ktt

#endif // KTT_API_CUDA
