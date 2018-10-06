#pragma once

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanQueue
{
public:
    VulkanQueue() :
        VulkanQueue(nullptr, VK_QUEUE_COMPUTE_BIT)
    {}

    explicit VulkanQueue(VkQueue queue, const VkQueueFlagBits queueType) :
        queue(queue),
        queueType(queueType)
    {}

    VkQueue getQueue() const
    {
        return queue;
    }

    VkQueueFlagBits getQueueType() const
    {
        return queueType;
    }

    void waitIdle()
    {
        vkQueueWaitIdle(queue);
    }

    void submitSingleCommand(VkCommandBuffer commandBuffer)
    {
        const VkSubmitInfo submitInfo =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &commandBuffer,
            0,
            nullptr
        };

        checkVulkanError(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
    }

private:
    VkQueue queue;
    VkQueueFlagBits queueType;
};

} // namespace ktt
