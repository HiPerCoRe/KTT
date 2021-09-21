#pragma once

#ifdef KTT_API_OPENCL

#include <map>
#include <memory>
#include <vector>

#include <Api/ComputeApiInitializer.h>
#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <ComputeEngine/ComputeEngine.h>
#include <ComputeEngine/EngineConfiguration.h>
#include <Utility/IdGenerator.h>
#include <Utility/LruCache.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaContext.h>
#include <ComputeEngine/OpenCl/Gpa/GpaInstance.h>
#include <ComputeEngine/OpenCl/Gpa/GpaInterface.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

class OpenClEngine : public ComputeEngine
{
public:
    explicit OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit OpenClEngine(const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds);

    // Kernel methods
    ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queueId) override;
    ComputationResult WaitForComputeAction(const ComputeActionId id) override;
    void ClearData(const KernelComputeId& id) override;
    void ClearKernelData(const std::string& kernelName) override;

    // Profiling methods
    ComputationResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) override;
    void SetProfilingCounters(const std::vector<std::string>& counters) override;
    bool IsProfilingSessionActive(const KernelComputeId& id) override;
    uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) override;
    bool HasAccurateRemainingProfilingRuns() const override;
    bool SupportsMultiInstanceProfiling() const override;

    // Buffer methods
    TransferActionId UploadArgument(KernelArgument& kernelArgument, const QueueId queueId) override;
    TransferActionId UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
        const size_t dataSize) override;
    TransferActionId DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
        const size_t dataSize) override;
    TransferActionId CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
        const size_t dataSize) override;
    TransferResult WaitForTransferAction(const TransferActionId id) override;
    void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) override;
    void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) override;
    void AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer) override;
    void ClearBuffer(const ArgumentId id) override;
    void ClearBuffers() override;
    bool HasBuffer(const ArgumentId id) override;

    // Queue methods
    QueueId AddComputeQueue(ComputeQueue queue) override;
    void RemoveComputeQueue(const QueueId id) override;
    QueueId GetDefaultQueue() const override;
    std::vector<QueueId> GetAllQueues() const override;
    void SynchronizeQueue(const QueueId queueId) override;
    void SynchronizeDevice() override;

    // Information retrieval methods
    std::vector<PlatformInfo> GetPlatformInfo() const override;
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platformIndex) const override;
    PlatformInfo GetCurrentPlatformInfo() const override;
    DeviceInfo GetCurrentDeviceInfo() const override;
    ComputeApi GetComputeApi() const override;
    GlobalSizeType GetGlobalSizeType() const override;

    // Utility methods
    void SetCompilerOptions(const std::string& options) override;
    void SetGlobalSizeType(const GlobalSizeType type) override;
    void SetAutomaticGlobalSizeCorrection(const bool flag) override;
    void SetKernelCacheCapacity(const uint64_t capacity) override;
    void ClearKernelCache() override;
    void EnsureThreadContext() override;

private:
    EngineConfiguration m_Configuration;
    PlatformIndex m_PlatformIndex;
    DeviceIndex m_DeviceIndex;
    DeviceInfo m_DeviceInfo;
    IdGenerator<QueueId> m_QueueIdGenerator;
    IdGenerator<ComputeActionId> m_ComputeIdGenerator;
    IdGenerator<TransferActionId> m_TransferIdGenerator;
    std::unique_ptr<OpenClContext> m_Context;
    std::map<QueueId, std::unique_ptr<OpenClCommandQueue>> m_Queues;
    std::map<ArgumentId, std::unique_ptr<OpenClBuffer>> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<OpenClKernel>> m_KernelCache;
    std::map<ComputeActionId, std::unique_ptr<OpenClComputeAction>> m_ComputeActions;
    std::map<TransferActionId, std::unique_ptr<OpenClTransferAction>> m_TransferActions;

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    std::unique_ptr<GpaInterface> m_GpaInterface;
    std::unique_ptr<GpaContext> m_GpaContext;
    std::map<KernelComputeId, std::unique_ptr<GpaInstance>> m_GpaInstances;
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

    std::shared_ptr<OpenClKernel> LoadKernel(const KernelComputeData& data);
    void SetKernelArguments(OpenClKernel& kernel, const std::vector<KernelArgument*> arguments);
    void SetKernelArgument(OpenClKernel& kernel, const KernelArgument& argument);
    std::unique_ptr<OpenClBuffer> CreateBuffer(KernelArgument& argument);
    std::unique_ptr<OpenClBuffer> CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer);

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    void InitializeGpa();
    void InitializeProfiling(const KernelComputeId& id);
    void FillProfilingData(const KernelComputeId& id, ComputationResult& result);
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
};

} // namespace ktt

#endif // KTT_API_OPENCL
