#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanCommandPool
{
public:
    explicit VulkanCommandPool(const VkDevice device, const uint32_t queueIndex) :
        device(device),
        queueIndex(queueIndex)
    {
        const VkCommandPoolCreateInfo commandPoolCreateInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            queueIndex
        };

        checkVulkanError(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool), "vkCreateCommandPool");
    }

    ~VulkanCommandPool()
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    uint32_t getQueueIndex() const
    {
        return queueIndex;
    }

    VkCommandPool getCommandPool() const
    {
        return commandPool;
    }

private:
    VkDevice device;
    uint32_t queueIndex;
    VkCommandPool commandPool;
};

} // namespace ktt
