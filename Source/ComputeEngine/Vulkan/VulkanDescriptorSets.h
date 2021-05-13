#pragma once

#ifdef KTT_API_VULKAN

#include <vector>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanBuffer;
class VulkanDescriptorPool;
class VulkanDescriptorSetLayout;
class VulkanDevice;  

class VulkanDescriptorSets
{
public:
    explicit VulkanDescriptorSets(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool,
        const std::vector<const VulkanDescriptorSetLayout*>& descriptorLayouts);
    ~VulkanDescriptorSets();

    VkDescriptorSet GetSet() const;
    const std::vector<VkDescriptorSet>& GetSets() const;

    void BindBuffer(const VulkanBuffer& buffer, const VkDescriptorType descriptorType, const size_t setIndex,
        const uint32_t binding);
    void BindBuffers(const std::vector<VulkanBuffer*>& buffers, const VkDescriptorType descriptorType, const size_t setIndex);

private:
    VkDevice m_Device;
    VkDescriptorPool m_Pool;
    std::vector<VkDescriptorSet> m_Sets;
};

} // namespace ktt

#endif // KTT_API_VULKAN
