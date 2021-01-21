#pragma once

#ifdef KTT_API_CUDA

#include <ComputeEngine/Cuda/Buffers/CudaBuffer.h>

namespace ktt
{

class CudaHostBuffer : public CudaBuffer
{
public:
    explicit CudaHostBuffer(KernelArgument& argument, ActionIdGenerator& generator);
    explicit CudaHostBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer);
    ~CudaHostBuffer() override;

    std::unique_ptr<CudaTransferAction> UploadData(const CudaStream& stream, const void* source,
        const size_t dataSize) override;
    std::unique_ptr<CudaTransferAction> DownloadData(const CudaStream& stream, void* destination,
        const size_t dataSize) const override;
    std::unique_ptr<CudaTransferAction> CopyData(const CudaStream& stream, const CudaBuffer& source,
        const size_t dataSize) override;
    void Resize(const size_t newSize, const bool preserveData) override;

protected:
    void* m_RawBuffer;
};

} // namespace ktt

#endif // KTT_API_CUDA
