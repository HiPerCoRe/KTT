#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanFence
{
public:
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
        vkDestroyFence(device, fence, nullptr);
    }

    void wait()
    {
        checkVulkanError(vkWaitForFences(device, 1, &fence, VK_TRUE, fenceTimeout), "vkWaitForFences");
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
    static const uint64_t fenceTimeout = 3600'000'000'000; // one hour
};

} // namespace ktt
