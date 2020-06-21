#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <cuda.h>
#include <ktt_types.h>

namespace ktt
{

class CUDAStream
{
public:
    explicit CUDAStream(const QueueId id);
    explicit CUDAStream(const QueueId id, UserQueue stream);
    ~CUDAStream();

    CUstream getStream() const;
    QueueId getId() const;

private:
    CUstream stream;
    QueueId id;
    bool owningStream;
};

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
