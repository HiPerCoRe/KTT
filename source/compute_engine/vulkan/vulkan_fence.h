#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanFence
{
public:
    explicit VulkanFence(const VkDevice device) :
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
        vkDestroyFence(device, fence, nullptr);
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
