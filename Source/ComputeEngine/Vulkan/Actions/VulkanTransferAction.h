#pragma once

#ifdef KTT_API_VULKAN

#include <memory>

#include <ComputeEngine/Vulkan/VulkanCommandBuffers.h>
#include <ComputeEngine/Vulkan/VulkanFence.h>
#include <ComputeEngine/TransferResult.h>
#include <KttTypes.h>

namespace ktt
{

class VulkanCommandPool;
class VulkanDevice;
class VulkanQueryPool;

class VulkanTransferAction
{
public:
    VulkanTransferAction(const TransferActionId id, const VulkanDevice* device = nullptr, const VulkanCommandPool* commandPool = nullptr,
        VulkanQueryPool* queryPool = nullptr);

    void SetDuration(const Nanoseconds duration);
    void IncreaseOverhead(const Nanoseconds overhead);
    void WaitForFinish();

    TransferActionId GetId() const;
    VkFence GetFence() const;
    VkCommandBuffer GetCommandBuffer() const;
    uint32_t GetFirstQueryId() const;
    uint32_t GetSecondQueryId() const;
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    bool IsAsync() const;
    TransferResult GenerateResult() const;

private:
    TransferActionId m_Id;
    std::unique_ptr<VulkanFence> m_Fence;
    std::unique_ptr<VulkanCommandBuffers> m_CommandBuffers;
    VulkanQueryPool* m_QueryPool;
    Nanoseconds m_Duration;
    Nanoseconds m_Overhead;
    uint32_t m_FirstQueryId;
    uint32_t m_SecondQueryId;
};

} // namespace ktt

#endif // KTT_API_VULKAN
