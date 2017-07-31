#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanBuffer
{
public:
    explicit VulkanBuffer(const VkDevice device, uint64_t bufferSize) :
        device(device),
        bufferSize(bufferSize)
    {
        /*const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            VkBufferUsageFlags usage,
            VK_SHARING_MODE_EXCLUSIVE,
            uint32_t queueFamilyIndexCount,
            const uint32_t* pQueueFamilyIndices
        };

        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");*/
    }

    ~VulkanBuffer()
    {
        vkDestroyBuffer(device, buffer, nullptr);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    uint64_t getBufferSize() const
    {
        return bufferSize;
    }

private:
    VkDevice device;
    VkBuffer buffer;
    uint64_t bufferSize;
};

} // namespace ktt
