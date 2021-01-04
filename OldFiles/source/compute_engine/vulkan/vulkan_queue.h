#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

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

    void submitSingleCommand(VkCommandBuffer commandBuffer) const
    {
        submitSingleCommand(commandBuffer, nullptr);
    }

    void submitSingleCommand(VkCommandBuffer commandBuffer, VkFence fence) const
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

        checkVulkanError(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit");
    }

private:
    VkQueue queue;
    VkQueueFlagBits queueType;
};

} // namespace ktt
