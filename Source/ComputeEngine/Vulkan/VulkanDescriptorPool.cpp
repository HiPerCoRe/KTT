#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanDescriptorPool.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorSetLayout.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanDescriptorPool::VulkanDescriptorPool(const VulkanDevice& device, const VkDescriptorType descriptorType,
    const uint32_t descriptorCount) :
    m_Device(device),
    m_DescriptorCount(descriptorCount)
{
    Logger::LogDebug("Initializing Vulkan descriptor pool");

    const VkDescriptorPoolSize poolSize =
    {
        descriptorType,
        m_DescriptorCount
    };

    const VkDescriptorPoolCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        m_DescriptorCount,
        1,
        &poolSize
    };

    CheckError(vkCreateDescriptorPool(m_Device.GetDevice(), &createInfo, nullptr, &m_Pool), "vkCreateDescriptorPool");
}

VulkanDescriptorPool::~VulkanDescriptorPool()
{
    Logger::LogDebug("Releasing Vulkan descriptor pool");
    vkDestroyDescriptorPool(m_Device.GetDevice(), m_Pool, nullptr);
}

std::unique_ptr<VulkanDescriptorSets> VulkanDescriptorPool::AllocateSets(const std::vector<const VulkanDescriptorSetLayout*>& layouts) const
{
    return std::make_unique<VulkanDescriptorSets>(m_Device, *this, layouts);
}

VkDescriptorPool VulkanDescriptorPool::GetPool() const
{
    return m_Pool;
}

} // namespace ktt

#endif // KTT_API_VULKAN
