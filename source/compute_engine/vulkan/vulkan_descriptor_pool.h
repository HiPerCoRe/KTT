#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDescriptorPool
{
public:
    VulkanDescriptorPool() :
        device(nullptr),
        descriptorPool(nullptr)
    {}

    explicit VulkanDescriptorPool(VkDevice device) :
        VulkanDescriptorPool(device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
    {}

    explicit VulkanDescriptorPool(VkDevice device, const VkDescriptorType descriptorType, const uint32_t descriptorCount) :
        device(device)
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
            1,
            1,
            &poolSize
        };

        checkVulkanError(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");
    }

    ~VulkanDescriptorPool()
    {
        if (descriptorPool != nullptr)
        {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkDescriptorPool getDescriptorPool() const
    {
        return descriptorPool;
    }

private:
    VkDevice device;
    VkDescriptorPool descriptorPool;
};

} // namespace ktt
