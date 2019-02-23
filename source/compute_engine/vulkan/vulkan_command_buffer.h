#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanCommandBuffer
{
public:
    explicit VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool) :
        VulkanCommandBuffer(device, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    {}

    explicit VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool, const VkCommandBufferLevel commandBufferLevel) :
        device(device),
        commandPool(commandPool),
        commandBufferLevel(commandBufferLevel)
    {
        const VkCommandBufferAllocateInfo commandBufferAllocateInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            commandBufferLevel,
            1
        };

        checkVulkanError(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer), "vkAllocateCommandBuffers");
    }

    ~VulkanCommandBuffer()
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkCommandPool getCommandPool() const
    {
        return commandPool;
    }

    VkCommandBuffer getCommandBuffer() const
    {
        return commandBuffer;
    }

    VkCommandBufferLevel getCommandBufferLevel() const
    {
        return commandBufferLevel;
    }

private:
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkCommandBufferLevel commandBufferLevel;
};

} // namespace ktt
