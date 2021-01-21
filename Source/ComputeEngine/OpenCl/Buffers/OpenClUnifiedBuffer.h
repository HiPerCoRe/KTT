#pragma once

#ifdef KTT_API_OPENCL
#ifdef CL_VERSION_2_0

#include <CL/cl.h>

#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>

namespace ktt
{

class OpenClUnifiedBuffer : public OpenClBuffer
{
public:
    explicit OpenClUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context);
    explicit OpenClUnifiedBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer);
    ~OpenClUnifiedBuffer() override;

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
    void* m_SvmBuffer;
};

} // namespace ktt

#endif // CL_VERSION_2_0
#endif // KTT_API_OPENCL
