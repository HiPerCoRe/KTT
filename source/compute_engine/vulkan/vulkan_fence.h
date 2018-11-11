#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanFence
{
public:
    VulkanFence() :
        device(nullptr),
        fence(nullptr)
    {}

    explicit VulkanFence(VkDevice device) :
        device(device)
    {
        const VkFenceCreateInfo fenceCreateInfo =
        {
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            nullptr,
            0
        };

        checkVulkanError(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence), "vkCreateFence");
    }

    ~VulkanFence()
    {
        if (fence != nullptr)
        {
            vkDestroyFence(device, fence, nullptr);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkFence getFence() const
    {
        return fence;
    }

private:
    VkDevice device;
    VkFence fence;
};

} // namespace ktt
