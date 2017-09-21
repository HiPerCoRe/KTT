#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanQueue
{
public:
    explicit VulkanQueue(const VkDevice device, const uint32_t queueIndex) :
        device(device),
        queueIndex(queueIndex)
    {
        vkGetDeviceQueue(device, queueIndex, 0, &queue);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    uint32_t getQueueIndex() const
    {
        return queueIndex;
    }

    VkQueue getQueue() const
    {
        return queue;
    }

private:
    VkDevice device;
    uint32_t queueIndex;
    VkQueue queue;
};

} // namespace ktt
