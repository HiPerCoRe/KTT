#pragma once

#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>

namespace ktt
{

class OpenClHostBuffer : public OpenClBuffer
{
public:
    explicit OpenClHostBuffer(KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context);
    explicit OpenClHostBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer);
    ~OpenClHostBuffer() override;

    std::unique_ptr<OpenClTransferAction> UploadData(const OpenClCommandQueue& queue, const void* source,
        const size_t dataSize) override;
    std::unique_ptr<OpenClTransferAction> DownloadData(const OpenClCommandQueue& queue, void* destination,
        const size_t dataSize) const override;
    std::unique_ptr<OpenClTransferAction> CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
        const size_t dataSize) override;
    void Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData) override;

    cl_mem GetBuffer() const override;
    void* GetRawBuffer() override;

private:
    cl_mem m_Buffer;
    void* m_RawBuffer;

    bool IsZeroCopy() const;
};

} // namespace ktt

#endif // KTT_API_OPENCL
