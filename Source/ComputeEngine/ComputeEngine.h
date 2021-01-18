#pragma once

#include <string>
#include <vector>

#include <Api/Info/DeviceInfo.h>
#include <Api/Info/PlatformInfo.h>
#include <Api/Output/KernelResult.h>
#include <ComputeEngine/KernelComputeData.h>
#include <ComputeEngine/GlobalSizeType.h>
#include <KernelArgument/KernelArgument.h>
#include <KttTypes.h>

namespace ktt
{

class ComputeEngine
{
public:
    virtual ~ComputeEngine() = default;

    // Kernel methods
    virtual ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queueId) = 0;
    virtual KernelResult WaitForComputeAction(const ComputeActionId id) = 0;

    // Profiling methods
    virtual KernelResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) = 0;
    virtual void SetProfilingCounters(const std::vector<std::string>& counters) = 0;
    virtual bool IsProfilingSessionActive(const KernelComputeId& id) = 0;
    virtual uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) = 0;
    virtual bool HasAccurateRemainingProfilingRuns() const = 0;

    // Buffer methods
    virtual TransferActionId UploadArgument(const KernelArgument& kernelArgument, const QueueId queueId) = 0;
    virtual TransferActionId UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
        const size_t dataSize) = 0;
    virtual TransferActionId DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
        const size_t dataSize) = 0;
    virtual TransferActionId CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
        const size_t dataSize) = 0;
    virtual uint64_t WaitForTransferAction(const TransferActionId id) = 0;
    virtual void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) = 0;
    virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) = 0;
    virtual void AddCustomBuffer(const KernelArgument& kernelArgument, ComputeBuffer buffer) = 0;
    virtual void ClearBuffer(const ArgumentId id) = 0;
    virtual void ClearBuffers() = 0;

    // Queue methods
    virtual QueueId GetDefaultQueue() const = 0;
    virtual std::vector<QueueId> GetAllQueues() const = 0;
    virtual void SynchronizeQueue(const QueueId queueId) = 0;
    virtual void SynchronizeDevice() = 0;

    // Information retrieval methods
    virtual std::vector<PlatformInfo> GetPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platformIndex) const = 0;
    virtual DeviceInfo GetCurrentDeviceInfo() const = 0;

    // Utility methods
    virtual void SetCompilerOptions(const std::string& options) = 0;
    virtual void SetGlobalSizeType(const GlobalSizeType type) = 0;
    virtual void SetAutomaticGlobalSizeCorrection(const bool flag) = 0;
    virtual void SetKernelCacheCapacity(const uint64_t capacity) = 0;
    virtual void ClearKernelCache() = 0;
};

} // namespace ktt
