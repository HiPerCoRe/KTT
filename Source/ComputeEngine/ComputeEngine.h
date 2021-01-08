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
    virtual ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queue, const bool runWithProfiling) = 0;
    virtual KernelResult WaitForComputeAction(const ComputeActionId id) const = 0;

    // Profiling methods
    virtual void SetProfilingCounters(const std::vector<std::string>& counters) = 0;
    virtual bool IsProfilingSessionActive(const KernelComputeId& id) = 0;
    virtual uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) = 0;
    virtual bool HasAccurateRemainingProfilingRuns() const = 0;

    // Buffer methods
    virtual TransferActionId UploadArgumentAsync(const KernelArgument& kernelArgument, const QueueId queue) = 0;
    virtual TransferActionId UpdateArgumentAsync(const ArgumentId id, const QueueId queue, const void* data, const size_t dataSize) = 0;
    virtual TransferActionId DownloadArgumentAsync(const ArgumentId id, const QueueId queue, void* destination, const size_t dataSize) const = 0;
    virtual TransferActionId CopyArgumentAsync(const ArgumentId destination, const QueueId queue, const ArgumentId source, const size_t dataSize) = 0;
    virtual uint64_t WaitForTransferAction(const TransferActionId id) const = 0;
    virtual void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) = 0;
    virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) = 0;
    virtual void AddCustomBuffer(const KernelArgument& kernelArgument, ComputeBuffer buffer) = 0;
    virtual void ClearBuffer(const ArgumentId id) = 0;
    virtual void ClearBuffers() = 0;

    // Queue methods
    virtual QueueId GetDefaultQueue() const = 0;
    virtual std::vector<QueueId> GetAllQueues() const = 0;
    virtual void SynchronizeQueue(const QueueId queue) = 0;
    virtual void SynchronizeDevice() = 0;

    // Information retrieval methods
    virtual std::vector<PlatformInfo> GetPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platform) const = 0;
    virtual DeviceInfo GetCurrentDeviceInfo() const = 0;

    // Utility methods
    virtual void SetCompilerOptions(const std::string& options) = 0;
    virtual void SetGlobalSizeType(const GlobalSizeType type) = 0;
    virtual void SetAutomaticGlobalSizeCorrection(const bool flag) = 0;
    virtual void SetKernelCacheCapacity(const size_t capacity) = 0;
    virtual void ClearKernelCache() = 0;
};

} // namespace ktt
