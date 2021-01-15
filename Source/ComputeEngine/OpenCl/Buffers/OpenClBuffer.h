#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <CL/cl.h>

#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <KernelArgument/ArgumentMemoryLocation.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClBuffer
{
public:
    virtual ~OpenClBuffer() = default;

    virtual std::unique_ptr<OpenClTransferAction> UploadData(const OpenClCommandQueue& queue, const void* source,
        const size_t dataSize) = 0;
    virtual std::unique_ptr<OpenClTransferAction> DownloadData(const OpenClCommandQueue& queue, void* destination,
        const size_t dataSize) const = 0;
    virtual std::unique_ptr<OpenClTransferAction> CopyData(const OpenClCommandQueue& queue, const OpenClBuffer& source,
        const size_t dataSize) = 0;
    virtual void Resize(const OpenClCommandQueue& queue, const size_t newSize, const bool preserveData) = 0;

    virtual ArgumentId GetArgumentId() const = 0;
    virtual ArgumentAccessType GetAccessType() const = 0;
    virtual ArgumentMemoryLocation GetMemoryLocation() const = 0;
    virtual cl_mem GetBuffer() const = 0;
    virtual void* GetRawBuffer() = 0;
    virtual size_t GetSize() const = 0;

protected:
    cl_mem_flags GetMemoryFlags();
};

} // namespace ktt

#endif // KTT_API_OPENCL
