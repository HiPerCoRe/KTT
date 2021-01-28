#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanDevice;

class VulkanFence
{
public:
    explicit VulkanFence(const VulkanDevice& device);
    ~VulkanFence();

    void Wait();
    void Wait(const uint64_t duration);

private:
    VkDevice m_Device;
    VkFence m_Fence;
};

} // namespace ktt

#endif // KTT_API_VULKAN
