#pragma once

#ifdef KTT_API_VULKAN

#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanDevice;

class VulkanSemaphore
{
public:
    explicit VulkanSemaphore(const VulkanDevice& device);
    ~VulkanSemaphore();

    VkSemaphore GetSemaphore() const;

private:
    VkDevice m_Device;
    VkSemaphore m_Semaphore;
};

} // namespace ktt

#endif // KTT_API_VULKAN
