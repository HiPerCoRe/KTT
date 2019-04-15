#pragma once

#ifdef KTT_PLATFORM_VULKAN

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_buffer.h>
#include <compute_engine/vulkan/vulkan_command_buffer_holder.h>
#include <compute_engine/vulkan/vulkan_command_pool.h>
#include <compute_engine/vulkan/vulkan_compute_pipeline.h>
#include <compute_engine/vulkan/vulkan_descriptor_pool.h>
#include <compute_engine/vulkan/vulkan_descriptor_set_holder.h>
#include <compute_engine/vulkan/vulkan_descriptor_set_layout.h>
#include <compute_engine/vulkan/vulkan_device.h>
#include <compute_engine/vulkan/vulkan_event.h>
#include <compute_engine/vulkan/vulkan_instance.h>
#include <compute_engine/vulkan/vulkan_pipeline_cache_entry.h>
#include <compute_engine/vulkan/vulkan_physical_device.h>
#include <compute_engine/vulkan/vulkan_query_pool.h>
#include <compute_engine/vulkan/vulkan_queue.h>
#include <compute_engine/vulkan/vulkan_semaphore.h>
#include <compute_engine/vulkan/vulkan_shader_module.h>
#include <compute_engine/vulkan/vulkan_utility.h>
#include <compute_engine/compute_engine.h>

namespace ktt
{

class VulkanEngine : public ComputeEngine
{
public:
    // Constructor
    explicit VulkanEngine(const DeviceIndex deviceIndex, const uint32_t queueCount);

    // Kernel handling methods
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<OutputDescriptor>& outputDescriptors) override;
    EventId runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const override;
    uint64_t getKernelOverhead(const EventId id) const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setKernelCacheUsage(const bool flag) override;
    void setKernelCacheCapacity(const size_t capacity) override;
    void clearKernelCache() override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;
    void clearEvents() override;

    // Argument handling methods
    uint64_t uploadArgument(KernelArgument& kernelArgument) override;
    EventId uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue) override;
    uint64_t updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    EventId updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    EventId downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const override;
    KernelArgument downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const override;
    uint64_t copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes) override;
    EventId copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t persistArgument(KernelArgument& kernelArgument, const bool flag) override;
    uint64_t getArgumentOperationDuration(const EventId id) const override;
    void resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) override;
    void setPersistentBufferUsage(const bool flag) override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType accessType) override;

    // Information retrieval methods
    void printComputeAPIInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Kernel profiling methods
    void initializeKernelProfiling(const KernelRuntimeData& kernelData) override;
    EventId runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const QueueId queue) override;
    uint64_t getRemainingKernelProfilingRuns(const std::string& kernelName, const std::string& kernelSource) override;
    KernelResult getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) override;
    void setKernelProfilingCounters(const std::vector<std::string>& counterNames) override;

private:
    // Attributes
    DeviceIndex deviceIndex;
    uint32_t queueCount;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    bool kernelCacheFlag;
    size_t kernelCacheCapacity;
    bool persistentBufferFlag;
    mutable EventId nextEventId;
    std::unique_ptr<VulkanInstance> instance;
    std::unique_ptr<VulkanDevice> device;
    std::unique_ptr<VulkanCommandPool> commandPool;
    std::unique_ptr<VulkanQueryPool> queryPool;
    std::vector<VulkanQueue> queues;
    std::set<std::unique_ptr<VulkanBuffer>> buffers;
    std::set<std::unique_ptr<VulkanBuffer>> persistentBuffers;
    std::map<std::pair<std::string, std::string>, std::unique_ptr<VulkanPipelineCacheEntry>> pipelineCache;
    mutable std::map<EventId, std::unique_ptr<VulkanEvent>> kernelEvents;
    mutable std::map<EventId, std::unique_ptr<VulkanEvent>> bufferEvents;
    mutable std::map<EventId, std::unique_ptr<VulkanCommandBufferHolder>> eventCommands;
    mutable std::map<EventId, std::unique_ptr<VulkanBuffer>> stagingBuffers;

    EventId enqueuePipeline(VulkanComputePipeline& pipeline, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const QueueId queue, const uint64_t kernelLaunchOverhead);
    KernelResult createKernelResult(const EventId id) const;
    std::vector<VulkanBuffer*> getPipelineArguments(const std::vector<KernelArgument*>& argumentPointers);
    VulkanBuffer* findBuffer(const ArgumentId id) const;
};

} // namespace ktt

#endif // KTT_PLATFORM_VULKAN
