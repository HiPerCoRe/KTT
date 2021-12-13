#pragma once

#ifdef KTT_API_VULKAN

#include <map>
#include <memory>
#include <vector>

#include <Api/ComputeApiInitializer.h>
#include <ComputeEngine/Vulkan/Actions/VulkanComputeAction.h>
#include <ComputeEngine/Vulkan/Actions/VulkanTransferAction.h>
#include <ComputeEngine/Vulkan/ShadercCompiler.h>
#include <ComputeEngine/Vulkan/VulkanBuffer.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanComputePipeline.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanInstance.h>
#include <ComputeEngine/Vulkan/VulkanMemoryAllocator.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <ComputeEngine/Vulkan/VulkanQueue.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/ComputeEngine.h>
#include <ComputeEngine/EngineConfiguration.h>
#include <Utility/IdGenerator.h>
#include <Utility/LruCache.h>

namespace ktt
{

class VulkanEngine : public ComputeEngine
{
public:
    explicit VulkanEngine(const DeviceIndex deviceIndex, const uint32_t queueCount);

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
    void SynchronizeQueues() override;
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
    DeviceIndex m_DeviceIndex;
    DeviceInfo m_DeviceInfo;
    IdGenerator<ComputeActionId> m_ComputeIdGenerator;
    IdGenerator<TransferActionId> m_TransferIdGenerator;
    std::unique_ptr<VulkanInstance> m_Instance;
    std::unique_ptr<VulkanDevice> m_Device;
    std::unique_ptr<VulkanCommandPool> m_CommandPool;
    std::unique_ptr<VulkanDescriptorPool> m_DescriptorPool;
    std::unique_ptr<VulkanQueryPool> m_QueryPool;
    std::unique_ptr<ShadercCompiler> m_Compiler;
    std::unique_ptr<VulkanMemoryAllocator> m_Allocator;
    std::vector<std::unique_ptr<VulkanQueue>> m_Queues;
    std::map<ArgumentId, std::unique_ptr<VulkanBuffer>> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<VulkanComputePipeline>> m_PipelineCache;
    std::map<ComputeActionId, std::unique_ptr<VulkanComputeAction>> m_ComputeActions;
    std::map<TransferActionId, std::unique_ptr<VulkanTransferAction>> m_TransferActions;

    std::shared_ptr<VulkanComputePipeline> LoadPipeline(const KernelComputeData& data);
    VulkanBuffer* GetPipelineArgument(KernelArgument& argument);
    std::vector<VulkanBuffer*> GetPipelineArguments(const std::vector<KernelArgument*>& arguments);
    std::unique_ptr<VulkanBuffer> CreateBuffer(KernelArgument& argument);
    std::unique_ptr<VulkanBuffer> CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer);
    void ClearQueueActions(const QueueId id);
    static std::vector<KernelArgument*> GetScalarArguments(const std::vector<KernelArgument*>& arguments);
};

} // namespace ktt

#endif // KTT_API_VULKAN
