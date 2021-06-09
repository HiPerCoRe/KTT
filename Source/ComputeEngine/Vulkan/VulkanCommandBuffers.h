#pragma once

#ifdef KTT_API_VULKAN

#include <vector>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanCommandPool;
class VulkanDevice;

class VulkanCommandBuffers
{
public:
    explicit VulkanCommandBuffers(const VulkanDevice& device, const VulkanCommandPool& commandPool, const uint32_t count);
    explicit VulkanCommandBuffers(const VulkanDevice& device, const VulkanCommandPool& commandPool, const uint32_t count,
        const VkCommandBufferLevel level);
    ~VulkanCommandBuffers();

    VkCommandBuffer GetBuffer() const;
    const std::vector<VkCommandBuffer>& GetBuffers() const;
    VkCommandBufferLevel GetLevel() const;

private:
    VkDevice m_Device;
    VkCommandPool m_Pool;
    std::vector<VkCommandBuffer> m_Buffers;
    VkCommandBufferLevel m_Level;
};

} // namespace ktt

#endif // KTT_API_VULKAN
