#ifdef KTT_API_VULKAN

#include <vector>

#include <ComputeEngine/Vulkan/VulkanDescriptorSetLayout.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(const VulkanDevice& device, const VkDescriptorType descriptorType,
    const uint32_t bindingCount) :
    m_Device(device.GetDevice()),
    m_Type(descriptorType),
    m_BindingCount(bindingCount)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);

    for (uint32_t i = 0; i < bindingCount; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = descriptorType;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    const VkDescriptorSetLayoutCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        bindingCount,
        bindings.data()
    };

    CheckError(vkCreateDescriptorSetLayout(m_Device, &createInfo, nullptr, &m_Layout), "vkCreateDescriptorSetLayout");
}

VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
{
    vkDestroyDescriptorSetLayout(m_Device, m_Layout, nullptr);
}

VkDescriptorSetLayout VulkanDescriptorSetLayout::GetLayout() const
{
    return m_Layout;
}

uint32_t VulkanDescriptorSetLayout::GetBindingCount() const
{
    return m_BindingCount;
}

} // namespace ktt

#endif // KTT_API_VULKAN
