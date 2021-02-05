/** @file ComputeInterface.h
  * ...
  */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class ComputeInterface
  * ...
  */
class KTT_API ComputeInterface
{
public:
    /** @fn virtual ~ComputeInterface()
      * Default destructor.
      */
    virtual ~ComputeInterface() = default;

    virtual void RunKernel(const KernelDefinitionId id) = 0;

    virtual void RunKernel(const KernelDefinitionId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue) = 0;

    virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue, const DimensionVector& globalSize,
        const DimensionVector& localSize) = 0;

    virtual void WaitForComputeAction(const ComputeActionId id) = 0;

    virtual void RunKernelWithProfiling(const KernelDefinitionId id) = 0;

    virtual void RunKernelWithProfiling(const KernelDefinitionId id, const DimensionVector& globalSize,
        const DimensionVector& localSize) = 0;

    virtual uint64_t GetRemainingProfilingRuns(const KernelDefinitionId id) const = 0;

    virtual QueueId GetDefaultQueue() const = 0;

    virtual std::vector<QueueId> GetAllQueues() const = 0;

    virtual void SynchronizeQueue(const QueueId queue) = 0;

    virtual void SynchronizeDevice() = 0;

    virtual const DimensionVector& GetCurrentGlobalSize(const KernelDefinitionId id) const = 0;

    virtual const DimensionVector& GetCurrentLocalSize(const KernelDefinitionId id) const = 0;

    virtual const KernelConfiguration& GetCurrentConfiguration() const = 0;

    virtual void ChangeArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& arguments) = 0;

    virtual void SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second) = 0;

    virtual void UpdateScalarArgument(const ArgumentId id, const void* data) = 0;

    virtual void UpdateLocalArgument(const ArgumentId id, const size_t dataSize) = 0;

    virtual TransferActionId UploadBuffer(const ArgumentId id, const QueueId queue) = 0;

    virtual TransferActionId DownloadBuffer(const ArgumentId id, const QueueId queue, void* destination, const size_t dataSize) = 0;

    virtual TransferActionId UpdateBuffer(const ArgumentId id, const QueueId queue, const void* data, const size_t dataSize) = 0;

    virtual TransferActionId CopyBuffer(const ArgumentId destination, const ArgumentId source, const QueueId queue,
        const size_t dataSize) = 0;

    virtual void WaitForTransferAction(const TransferActionId id) = 0;

    virtual void ResizeBuffer(const ArgumentId id, const size_t newDataSize, const bool preserveData) = 0;

    virtual void ClearBuffer(const ArgumentId id) = 0;

    virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& memoryHandle) = 0;
};

} // namespace ktt
