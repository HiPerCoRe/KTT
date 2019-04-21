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

    explicit VulkanDescriptorSetLayout(VkDevice device, const VkDescriptorType descriptorType, const uint32_t bindingCount) :
        device(device),
        descriptorType(descriptorType),
        bindingCount(bindingCount)
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

        const VkDescriptorSetLayoutCreateInfo layoutCreateInfo =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            bindingCount,
            bindings.data()
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

    uint32_t getBindingCount() const
    {
        return bindingCount;
    }

private:
    VkDevice device;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorType descriptorType;
    uint32_t bindingCount;
};

} // namespace ktt
