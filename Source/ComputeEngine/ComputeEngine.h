#pragma once

#include <string>
#include <vector>

#include <Api/Info/DeviceInfo.h>
#include <Api/Info/PlatformInfo.h>
#include <Api/Output/ComputationResult.h>
#include <ComputeEngine/ComputeApi.h>
#include <ComputeEngine/KernelComputeData.h>
#include <ComputeEngine/GlobalSizeType.h>
#include <ComputeEngine/TransferResult.h>
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
    virtual ComputationResult WaitForComputeAction(const ComputeActionId id) = 0;
    virtual void ClearData(const KernelComputeId& id) = 0;
    virtual void ClearKernelData(const std::string& kernelName) = 0;

    // Profiling methods
    virtual ComputationResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) = 0;
    virtual void SetProfilingCounters(const std::vector<std::string>& counters) = 0;
    virtual bool IsProfilingSessionActive(const KernelComputeId& id) = 0;
    virtual uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) = 0;
    virtual bool HasAccurateRemainingProfilingRuns() const = 0;
    virtual bool SupportsMultiInstanceProfiling() const = 0;

    // Buffer methods
    virtual TransferActionId UploadArgument(KernelArgument& kernelArgument, const QueueId queueId) = 0;
    virtual TransferActionId UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
        const size_t dataSize) = 0;
    virtual TransferActionId DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
        const size_t dataSize) = 0;
    virtual TransferActionId CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
        const size_t dataSize) = 0;
    virtual TransferResult WaitForTransferAction(const TransferActionId id) = 0;
    virtual void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) = 0;
    virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) = 0;
    virtual void AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer) = 0;
    virtual void ClearBuffer(const ArgumentId id) = 0;
    virtual void ClearBuffers() = 0;
    virtual bool HasBuffer(const ArgumentId id) = 0;

    // Queue methods
    virtual QueueId AddComputeQueue(ComputeQueue queue) = 0;
    virtual void RemoveComputeQueue(const QueueId id) = 0;
    virtual QueueId GetDefaultQueue() const = 0;
    virtual std::vector<QueueId> GetAllQueues() const = 0;
    virtual void SynchronizeQueue(const QueueId queueId) = 0;
    virtual void SynchronizeQueues() = 0;
    virtual void SynchronizeDevice() = 0;

    // Information retrieval methods
    virtual std::vector<PlatformInfo> GetPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platformIndex) const = 0;
    virtual PlatformInfo GetCurrentPlatformInfo() const = 0;
    virtual DeviceInfo GetCurrentDeviceInfo() const = 0;
    virtual ComputeApi GetComputeApi() const = 0;
    virtual GlobalSizeType GetGlobalSizeType() const = 0;

    // Utility methods
    virtual void SetCompilerOptions(const std::string& options, const bool overrideDefault = false) = 0;
    virtual void SetGlobalSizeType(const GlobalSizeType type) = 0;
    virtual void SetAutomaticGlobalSizeCorrection(const bool flag) = 0;
    virtual void SetKernelCacheCapacity(const uint64_t capacity) = 0;
    virtual void ClearKernelCache() = 0;
    virtual void EnsureThreadContext() = 0;
};

} // namespace ktt
