#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanSemaphore.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

VulkanSemaphore::VulkanSemaphore(const VulkanDevice& device) :
    m_Device(device.GetDevice())
{
    const VkSemaphoreCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        nullptr,
        0
    };

    CheckError(vkCreateSemaphore(m_Device, &createInfo, nullptr, &m_Semaphore), "vkCreateSemaphore");
}

VulkanSemaphore::~VulkanSemaphore()
{
    vkDestroySemaphore(m_Device, m_Semaphore, nullptr);
}

VkSemaphore VulkanSemaphore::GetSemaphore() const
{
    return m_Semaphore;
}

} // namespace ktt

#endif // KTT_API_VULKAN
