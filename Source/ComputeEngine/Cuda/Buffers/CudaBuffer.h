#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <cuda.h>

#include <KernelArgument/ArgumentAccessType.h>
#include <KernelArgument/ArgumentMemoryLocation.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class CudaStream;
class CudaTransferAction;

class CudaBuffer
{
public:
    explicit CudaBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator);
    explicit CudaBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator, ComputeBuffer userBuffer);
    virtual ~CudaBuffer() = default;

    virtual std::unique_ptr<CudaTransferAction> UploadData(const CudaStream& stream, const void* source,
        const size_t dataSize) = 0;
    virtual std::unique_ptr<CudaTransferAction> DownloadData(const CudaStream& stream, void* destination,
        const size_t dataSize) const = 0;
    virtual std::unique_ptr<CudaTransferAction> CopyData(const CudaStream& stream, const CudaBuffer& source,
        const size_t dataSize) = 0;
    virtual void Resize(const size_t newSize, const bool preserveData) = 0;

    const CUdeviceptr* GetBuffer() const;
    CUdeviceptr* GetBuffer();
    ArgumentId GetArgumentId() const;
    ArgumentAccessType GetAccessType() const;
    ArgumentMemoryLocation GetMemoryLocation() const;
    size_t GetSize() const;

protected:
    KernelArgument& m_Argument;
    IdGenerator<TransferActionId>& m_Generator;
    size_t m_BufferSize;
    CUdeviceptr m_Buffer;
    bool m_UserOwned;
};

} // namespace ktt

#endif // KTT_API_CUDA
