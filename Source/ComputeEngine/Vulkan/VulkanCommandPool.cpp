#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanCommandPool::VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex) :
    VulkanCommandPool(device, queueFamilyIndex, 0)
{}

VulkanCommandPool::VulkanCommandPool(const VulkanDevice& device, const uint32_t queueFamilyIndex,
    const VkCommandPoolCreateFlags flags) :
    m_Device(device)
{
    Logger::LogDebug("Initializing Vulkan command pool");

    const VkCommandPoolCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        flags,
        queueFamilyIndex
    };

    CheckError(vkCreateCommandPool(m_Device.GetDevice(), &createInfo, nullptr, &m_Pool), "vkCreateCommandPool");
}

VulkanCommandPool::~VulkanCommandPool()
{
    Logger::LogDebug("Releasing Vulkan command pool");
    vkDestroyCommandPool(m_Device.GetDevice(), m_Pool, nullptr);
}

std::unique_ptr<VulkanCommandBuffers> VulkanCommandPool::AllocateBuffers(const uint32_t count,
    const VkCommandBufferLevel level) const
{
    return std::make_unique<VulkanCommandBuffers>(m_Device, *this, count, level);
}

VkCommandPool VulkanCommandPool::GetPool() const
{
    return m_Pool;
}

} // namespace ktt

#endif // KTT_API_VULKAN
