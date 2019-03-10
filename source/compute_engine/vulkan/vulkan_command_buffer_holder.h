#pragma once

#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanCommandBufferHolder
{
public:
    explicit VulkanCommandBufferHolder(VkDevice device, VkCommandPool commandPool) :
        VulkanCommandBufferHolder(device, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    {}

    explicit VulkanCommandBufferHolder(VkDevice device, VkCommandPool commandPool, const VkCommandBufferLevel commandBufferLevel,
        const uint32_t commandBufferCount) :
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
            commandBufferCount
        };

        commandBuffers.resize(static_cast<size_t>(commandBufferCount));
        checkVulkanError(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()), "vkAllocateCommandBuffers");
    }

    ~VulkanCommandBufferHolder()
    {
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
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
        if (commandBuffers.empty())
        {
            throw std::runtime_error("Cannot retrieve command buffer from empty holder");
        }

        return commandBuffers[0];
    }

    const std::vector<VkCommandBuffer>& getCommandBuffers() const
    {
        return commandBuffers;
    }

    VkCommandBufferLevel getCommandBufferLevel() const
    {
        return commandBufferLevel;
    }

private:
    VkDevice device;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    VkCommandBufferLevel commandBufferLevel;
};

} // namespace ktt
