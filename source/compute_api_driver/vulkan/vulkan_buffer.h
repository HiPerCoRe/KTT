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
    explicit VulkanBuffer(const VkDevice device, const uint32_t queueIndex, const uint64_t bufferSize) :
        device(device),
        queueIndex(queueIndex),
        bufferSize(bufferSize)
    {
        const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueIndex
        };

        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");
    }

    ~VulkanBuffer()
    {
        vkDestroyBuffer(device, buffer, nullptr);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    uint32_t getQueueIndex() const
    {
        return queueIndex;
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
    uint32_t queueIndex;
    VkBuffer buffer;
    uint64_t bufferSize;
};

} // namespace ktt
