#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanDevice;

class VulkanCommandPool
{
public:
    explicit VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex);
    explicit VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlags flags);
    ~VulkanCommandPool();

    VkCommandPool GetPool() const;

private:
    VkDevice m_Device;
    VkCommandPool m_Pool;
};

} // namespace ktt

#endif // KTT_API_VULKAN
