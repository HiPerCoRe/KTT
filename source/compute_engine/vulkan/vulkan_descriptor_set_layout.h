#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDescriptorSetLayout
{
public:
    VulkanDescriptorSetLayout() :
        device(nullptr),
        descriptorSetLayout(nullptr)
    {}

    explicit VulkanDescriptorSetLayout(VkDevice device, const VkDescriptorType descriptorType, const uint32_t descriptorCount) :
        device(device),
        descriptorType(descriptorType),
        descriptorCount(descriptorCount)
    {
        const VkDescriptorSetLayoutBinding descriptorSetLayoutBinding =
        {
            0,
            descriptorType,
            descriptorCount,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr
        };

        const VkDescriptorSetLayoutCreateInfo layoutCreateInfo =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayoutBinding
        };

        checkVulkanError(vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &descriptorSetLayout), "vkCreateDescriptorSetLayout");
    }

    ~VulkanDescriptorSetLayout()
    {
        if (descriptorSetLayout != nullptr)
        {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkDescriptorSetLayout getDescriptorSetLayout() const
    {
        return descriptorSetLayout;
    }

    VkDescriptorType getDescriptorType() const
    {
        return descriptorType;
    }

    uint32_t getDescriptorCount() const
    {
        return descriptorCount;
    }

private:
    VkDevice device;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
};

} // namespace ktt
