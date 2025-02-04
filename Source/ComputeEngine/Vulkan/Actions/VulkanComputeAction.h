#pragma once

#ifdef KTT_API_VULKAN

#include <memory>

#include <Api//Configuration/DimensionVector.h>
#include <Api/Output/ComputationResult.h>
#include <ComputeEngine/Vulkan/VulkanComputePipeline.h>
#include <ComputeEngine/Vulkan/VulkanFence.h>
#include <KttTypes.h>

namespace ktt
{

class VulkanCommandPool;
class VulkanDevice;
class VulkanQueryPool;

class VulkanComputeAction
{
public:
    VulkanComputeAction(const ComputeActionId id, const QueueId queueId, const VulkanDevice& device,
        const VulkanCommandPool& commandPool, VulkanQueryPool& queryPool, std::shared_ptr<VulkanComputePipeline> pipeline,
        const DimensionVector& globalSize, const DimensionVector& localSize);

    void IncreaseOverhead(const Nanoseconds overhead);
    void IncreaseCompilationOverhead(const Nanoseconds overhead);
    void SetComputeId(const KernelComputeId& id);
    void WaitForFinish();

    ComputeActionId GetId() const;
    QueueId GetQueueId() const;
    VulkanComputePipeline& GetPipeline();
    VkFence GetFence() const;
    VkCommandBuffer GetCommandBuffer() const;
    uint32_t GetFirstQueryId() const;
    uint32_t GetSecondQueryId() const;
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    Nanoseconds GetCompilationOverhead() const;
    const KernelComputeId& GetComputeId() const;
    ComputationResult GenerateResult() const;

private:
    ComputeActionId m_Id;
    QueueId m_QueueId;
    std::shared_ptr<VulkanComputePipeline> m_Pipeline;
    std::unique_ptr<VulkanFence> m_Fence;
    std::unique_ptr<VulkanCommandBuffers> m_CommandBuffers;
    VulkanQueryPool& m_QueryPool;
    Nanoseconds m_Overhead;
    Nanoseconds m_CompilationOverhead;
    KernelComputeId m_ComputeId;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    uint32_t m_FirstQueryId;
    uint32_t m_SecondQueryId;
};

} // namespace ktt

#endif // KTT_API_VULKAN
