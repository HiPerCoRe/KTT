#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanBuffer
{
public:
    explicit VulkanBuffer(const VkDevice device, const uint32_t queueIndex, const VkDeviceSize bufferSize,
        const VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties) :
        device(device),
        queueIndex(queueIndex),
        bufferSize(bufferSize),
        bufferPrepared(false)
    {
        uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

        for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
        {
            if (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags
                && VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags
                && bufferSize <= physicalDeviceMemoryProperties.memoryHeaps[physicalDeviceMemoryProperties.memoryTypes[i].heapIndex].size)
            {
                memoryTypeIndex = i;
                break;
            }
        }

        const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            bufferSize,
            memoryTypeIndex
        };

        checkVulkanError(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &deviceMemory), "vkAllocateMemory");
    }

    ~VulkanBuffer()
    {
        if (bufferPrepared)
        {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        vkFreeMemory(device, deviceMemory, nullptr);
    }

    void uploadData(const void* source, const size_t dataSize, const VkCommandBuffer commandBuffer)
    {
        if (bufferSize != dataSize)
        {
            // to do: implement buffer resize
            throw std::runtime_error("Size of data to upload is higher than size of buffer");
        }

        void* mappedObject;
        checkVulkanError(vkMapMemory(device, deviceMemory, 0, dataSize, 0, &mappedObject), "vkMapMemory");
        std::memcpy(mappedObject, source, dataSize);
        vkUnmapMemory(device, deviceMemory);
    }

    void downloadData(void* destination, const size_t dataSize) const
    {
        if (bufferSize < dataSize)
        {
            // to do: implement buffer resize
            throw std::runtime_error("Size of data to download is higher than size of buffer");
        }

        void* mappedObject;
        checkVulkanError(vkMapMemory(device, deviceMemory, 0, dataSize, 0, &mappedObject), "vkMapMemory");
        std::memcpy(destination, mappedObject, dataSize);
        vkUnmapMemory(device, deviceMemory);
    }

    void prepareBuffer()
    {
        if (bufferPrepared)
        {
            return;
        }

        const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueIndex
        };
        
        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");
        checkVulkanError(vkBindBufferMemory(device, buffer, deviceMemory, 0), "vkBindBufferMemory");
        bufferPrepared = true;
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

    VkDeviceSize getBufferSize() const
    {
        return bufferSize;
    }

    VkDeviceMemory getDeviceMemory() const
    {
        return deviceMemory;
    }

private:
    VkDevice device;
    uint32_t queueIndex;
    VkBuffer buffer;
    VkDeviceSize bufferSize;
    VkDeviceMemory deviceMemory;
    bool bufferPrepared;
};

} // namespace ktt
