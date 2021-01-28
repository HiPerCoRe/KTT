#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanFence.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

VulkanFence::VulkanFence(const VulkanDevice& device) :
    m_Device(device.GetDevice())
{
    const VkFenceCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        0
    };

    CheckError(vkCreateFence(m_Device, &createInfo, nullptr, &m_Fence), "vkCreateFence");
}

VulkanFence::~VulkanFence()
{
    vkDestroyFence(m_Device, m_Fence, nullptr);
}

void VulkanFence::Wait()
{
    const uint64_t defaultDuration = 3600'000'000'000; // one hour
    Wait(defaultDuration);
}

void VulkanFence::Wait(const uint64_t duration)
{
    CheckError(vkWaitForFences(m_Device, 1, &m_Fence, VK_TRUE, duration), "vkWaitForFences");
}

} // namespace ktt

#endif // KTT_API_VULKAN
