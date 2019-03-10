#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDescriptorPool
{
public:
    explicit VulkanDescriptorPool(VkDevice device, const VkDescriptorType descriptorType, const uint32_t descriptorCount) :
        device(device),
        descriptorCount(descriptorCount)
    {
        const VkDescriptorPoolSize poolSize =
        {
            descriptorType,
            descriptorCount
        };

        const VkDescriptorPoolCreateInfo poolCreateInfo =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            descriptorCount,
            1,
            &poolSize
        };

        checkVulkanError(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");
    }

    ~VulkanDescriptorPool()
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkDescriptorPool getDescriptorPool() const
    {
        return descriptorPool;
    }

    uint32_t getDescriptorCount() const
    {
        return descriptorCount;
    }

private:
    VkDevice device;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorCount;
};

} // namespace ktt
