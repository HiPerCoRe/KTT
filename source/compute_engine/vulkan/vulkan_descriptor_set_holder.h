#pragma once

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_buffer.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDescriptorSetHolder
{
public:
    explicit VulkanDescriptorSetHolder(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout) :
        VulkanDescriptorSetHolder(device, descriptorPool, descriptorSetLayout, 1)
    {}

    explicit VulkanDescriptorSetHolder(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout,
        const uint32_t descriptorCount) :
        device(device),
        pool(descriptorPool)
    {
        const VkDescriptorSetAllocateInfo allocateInfo =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptorPool,
            1,
            &descriptorSetLayout
        };

        descriptorSets.resize(static_cast<size_t>(descriptorCount));
        checkVulkanError(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSets.data()), "vkAllocateDescriptorSets");
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkDescriptorSet getDescriptorSet() const
    {
        if (descriptorSets.empty())
        {
            throw std::runtime_error("Cannot retrieve descriptor set from empty holder");
        }

        return descriptorSets[0];
    }

    const std::vector<VkDescriptorSet>& getDescriptorSets() const
    {
        return descriptorSets;
    }

    void bindBuffer(const VulkanBuffer& buffer, const VkDescriptorType descriptorType, const size_t descriptorSetIndex)
    {
        bindBuffer(buffer, descriptorType, descriptorSetIndex, 0);
    }

    void bindBuffer(const VulkanBuffer& buffer, const VkDescriptorType descriptorType, const size_t descriptorSetIndex, const uint32_t binding)
    {
        if (descriptorSets.size() <= static_cast<size_t>(descriptorSetIndex))
        {
            throw std::runtime_error("Descriptor set index is out of range for this descriptor set holder");
        }

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
            descriptorSets[descriptorSetIndex],
            binding,
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
    VkDescriptorPool pool;
    std::vector<VkDescriptorSet> descriptorSets;
};

} // namespace ktt
