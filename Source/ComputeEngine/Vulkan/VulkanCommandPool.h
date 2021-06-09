#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/VulkanCommandBuffers.h>

namespace ktt
{

class VulkanDevice;

class VulkanCommandPool
{
public:
    explicit VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex);
    explicit VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlags flags);
    ~VulkanCommandPool();

    std::unique_ptr<VulkanCommandBuffers> AllocateBuffers(const uint32_t count,
        const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;
    VkCommandPool GetPool() const;

private:
    const VulkanDevice& m_Device;
    VkCommandPool m_Pool;
};

} // namespace ktt

#endif // KTT_API_VULKAN
