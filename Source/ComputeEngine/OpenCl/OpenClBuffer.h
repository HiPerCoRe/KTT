#pragma once

#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/ActionIdGenerator.h>
#include <KernelArgument/KernelArgument.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClBuffer
{
public:
    explicit OpenClBuffer(KernelArgument& argument, ActionIdGenerator& generator, const OpenClContext& context);
    explicit OpenClBuffer(KernelArgument& argument, ActionIdGenerator& generator, ComputeBuffer userBuffer);
    ~OpenClBuffer();

    std::unique_ptr<OpenClTransferAction> UploadData(const OpenClCommandQueue& queue, const void* source,
        const size_t dataSize);
    std::unique_ptr<OpenClTransferAction> DownloadData(const OpenClCommandQueue& queue, void* destination,
        const size_t dataSize) const;
    std::unique_ptr<OpenClTransferAction> CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
        const size_t dataSize);
    void Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData);

    KernelArgument& GetArgument() const;
    cl_context GetContext() const;
    cl_mem GetBuffer() const;
    void* GetRawBuffer() const;
    size_t GetSize() const;
    bool IsZeroCopy() const;

private:
    KernelArgument& m_Argument;
    ActionIdGenerator& m_Generator;
    cl_context m_Context;
    cl_mem m_Buffer;
    void* m_RawBuffer;
    size_t m_BufferSize;
    cl_mem_flags m_MemoryFlags;
    bool m_UserOwned;

    cl_mem_flags GetMemoryFlags();
};

} // namespace ktt

#endif // KTT_API_OPENCL
