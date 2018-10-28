#pragma once

#include <vector>
#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanCommandBufferGroup
{
public:
    VulkanCommandBufferGroup() :
        device(nullptr),
        commandPool(nullptr),
        commandBufferLevel(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    {}

    explicit VulkanCommandBufferGroup(VkDevice device, VkCommandPool commandPool, const uint32_t commandBufferCount) :
        VulkanCommandBufferGroup(device, commandPool, commandBufferCount, VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    {}

    explicit VulkanCommandBufferGroup(VkDevice device, VkCommandPool commandPool, const uint32_t commandBufferCount,
        const VkCommandBufferLevel commandBufferLevel) :
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

        commandBuffers.resize(commandBufferCount);
        checkVulkanError(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()), "vkAllocateCommandBuffers");
    }

    ~VulkanCommandBufferGroup()
    {
        if (commandPool != nullptr)
        {
            vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
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

    const std::vector<VkCommandBuffer>& getCommandBuffers() const
    {
        return commandBuffers;
    }

    uint32_t getCommandBufferCount()
    {
        return static_cast<uint32_t>(commandBuffers.size());
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
