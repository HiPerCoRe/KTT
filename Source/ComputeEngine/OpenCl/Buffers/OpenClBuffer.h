#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <CL/cl.h>

#include <KernelArgument/ArgumentAccessType.h>
#include <KernelArgument/ArgumentMemoryLocation.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClCommandQueue;
class OpenClContext;
class OpenClTransferAction;

class OpenClBuffer
{
public:
    explicit OpenClBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator, const OpenClContext& context);
    explicit OpenClBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator);
    virtual ~OpenClBuffer() = default;

    virtual std::unique_ptr<OpenClTransferAction> UploadData(const OpenClCommandQueue& queue, const void* source,
        const size_t dataSize) = 0;
    virtual std::unique_ptr<OpenClTransferAction> DownloadData(const OpenClCommandQueue& queue, void* destination,
        const size_t dataSize) const = 0;
    virtual std::unique_ptr<OpenClTransferAction> CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
        const size_t dataSize) = 0;
    virtual void Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData) = 0;

    virtual cl_mem GetBuffer() const = 0;
    virtual void* GetRawBuffer() = 0;

    ArgumentId GetArgumentId() const;
    ArgumentAccessType GetAccessType() const;
    ArgumentMemoryLocation GetMemoryLocation() const;
    size_t GetSize() const;

protected:
    KernelArgument& m_Argument;
    IdGenerator<TransferActionId>& m_Generator;
    cl_context m_Context;
    size_t m_BufferSize;
    cl_mem_flags m_MemoryFlags;
    bool m_UserOwned;

    cl_mem_flags GetMemoryFlags();
};

} // namespace ktt

#endif // KTT_API_OPENCL
