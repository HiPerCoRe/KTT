#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanCommandPool
{
public:
    VulkanCommandPool() :
        device(nullptr),
        commandPool(nullptr)
    {}

    explicit VulkanCommandPool(VkDevice device, const uint32_t queueFamilyIndex) :
        VulkanCommandPool(device, queueFamilyIndex, 0)
    {}

    explicit VulkanCommandPool(VkDevice device, const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlags commandPoolCreateFlags) :
        device(device)
    {
        const VkCommandPoolCreateInfo commandPoolCreateInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            commandPoolCreateFlags,
            queueFamilyIndex
        };

        checkVulkanError(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool), "vkCreateCommandPool");
    }

    ~VulkanCommandPool()
    {
        if (commandPool != nullptr)
        {
            vkDestroyCommandPool(device, commandPool, nullptr);
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

private:
    VkDevice device;
    VkCommandPool commandPool;
};

} // namespace ktt
