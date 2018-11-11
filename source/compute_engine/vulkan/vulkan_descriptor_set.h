#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_buffer.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDescriptorSet
{
public:
    VulkanDescriptorSet() :
        device(nullptr),
        descriptorSet(nullptr)
    {}

    explicit VulkanDescriptorSet(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout) :
        VulkanDescriptorSet(device, descriptorPool, descriptorSetLayout, 1)
    {}

    explicit VulkanDescriptorSet(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout,
        const uint32_t descriptorCount) :
        device(device)
    {
        const VkDescriptorSetAllocateInfo allocateInfo =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptorPool,
            descriptorCount,
            &descriptorSetLayout
        };

        checkVulkanError(vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet), "vkAllocateDescriptorSets");
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkDescriptorSet getDescriptorSet() const
    {
        return descriptorSet;
    }

    void bindBuffer(const VulkanBuffer& buffer, const VkDescriptorType descriptorType)
    {
        const VkDescriptorBufferInfo bufferInfo =
        {
            buffer.getBuffer(),
            0,
            buffer.getBufferSize()
        };

        const VkWriteDescriptorSet descriptorWrite =
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptorSet,
            0,
            0,
            1,
            descriptorType,
            nullptr,
            &bufferInfo,
            nullptr
        };

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }

private:
    VkDevice device;
    VkDescriptorSet descriptorSet;
};

} // namespace ktt
