#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanCommandBuffer
{
public:
    explicit VulkanCommandBuffer(const VkDevice device, const VkCommandPool commandPool) :
        device(device),
        commandPool(commandPool)
    {
        const VkCommandBufferAllocateInfo commandBufferAllocateInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
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

private:
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
};

} // namespace ktt
