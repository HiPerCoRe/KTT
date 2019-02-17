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
        fence(nullptr),
        id(0)
    {}

    explicit VulkanFence(VkDevice device, const EventId id) :
        device(device),
        id(id)
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

    EventId getId() const
    {
        return id;
    }

private:
    VkDevice device;
    VkFence fence;
    EventId id;
    static const uint64_t fenceTimeout = 3600'000'000'000;
};

} // namespace ktt
