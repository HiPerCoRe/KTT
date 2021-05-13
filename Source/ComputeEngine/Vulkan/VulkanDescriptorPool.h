#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/VulkanDescriptorSets.h>

namespace ktt
{

class VulkanDescriptorSetLayout;
class VulkanDevice;

class VulkanDescriptorPool
{
public:
    explicit VulkanDescriptorPool(const VulkanDevice& device, const VkDescriptorType descriptorType,
        const uint32_t descriptorCount);
    ~VulkanDescriptorPool();

    std::unique_ptr<VulkanDescriptorSets> AllocateSets(const std::vector<const VulkanDescriptorSetLayout*>& layouts) const;
    VkDescriptorPool GetPool() const;

private:
    const VulkanDevice& m_Device;
    VkDescriptorPool m_Pool;
    uint32_t m_DescriptorCount;
};

} // namespace ktt

#endif // KTT_API_VULKAN
