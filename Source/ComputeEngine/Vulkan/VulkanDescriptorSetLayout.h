#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanDevice;

class VulkanDescriptorSetLayout
{
public:
    explicit VulkanDescriptorSetLayout(const VulkanDevice& device, const VkDescriptorType descriptorType,
        const uint32_t bindingCount);
    ~VulkanDescriptorSetLayout();

    VkDescriptorSetLayout GetLayout() const;
    uint32_t GetBindingCount() const;

private:
    VkDevice m_Device;
    VkDescriptorSetLayout m_Layout;
    VkDescriptorType m_Type;
    uint32_t m_BindingCount;
};

} // namespace ktt

#endif // KTT_API_VULKAN
