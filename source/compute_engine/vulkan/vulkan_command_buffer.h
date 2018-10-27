#pragma once

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanCommandBuffer
{
public:
    VulkanCommandBuffer() :
        device(nullptr),
        commandPool(nullptr),
        commandBufferLevel(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    {}

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
        if (commandPool != nullptr)
        {
            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkCommandPool getCommandPool() const
    {
        return commandPool;
    }

    const VkCommandBuffer getCommandBuffer() const
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
