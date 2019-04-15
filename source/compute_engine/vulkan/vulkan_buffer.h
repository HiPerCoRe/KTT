#pragma once

#include <cstring>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_physical_device.h>
#include <compute_engine/vulkan/vulkan_utility.h>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

class VulkanBuffer
{
public:
    explicit VulkanBuffer(const VulkanBuffer& source, VkDevice device, const VulkanPhysicalDevice& physicalDevice,
        const VkBufferUsageFlags usageFlags, const VkDeviceSize bufferSize) :
        kernelArgumentId(source.getKernelArgumentId()),
        bufferSize(bufferSize),
        elementSize(source.getElementSize()),
        dataType(source.getDataType()),
        memoryLocation(ArgumentMemoryLocation::Host),
        accessType(source.getAccessType()),
        device(device),
        physicalDevice(&physicalDevice),
        bufferMemory(nullptr),
        usageFlags(usageFlags)
    {
        const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            usageFlags,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr
        };

        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");
    }

    explicit VulkanBuffer(KernelArgument& kernelArgument, VkDevice device, const VulkanPhysicalDevice& physicalDevice,
        const VkBufferUsageFlags usageFlags) :
        kernelArgumentId(kernelArgument.getId()),
        bufferSize(static_cast<VkDeviceSize>(kernelArgument.getDataSizeInBytes())),
        elementSize(kernelArgument.getElementSizeInBytes()),
        dataType(kernelArgument.getDataType()),
        memoryLocation(kernelArgument.getMemoryLocation()),
        accessType(kernelArgument.getAccessType()),
        device(device),
        physicalDevice(&physicalDevice),
        bufferMemory(nullptr),
        usageFlags(usageFlags)
    {
        const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            usageFlags,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr
        };

        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");
    }

    ~VulkanBuffer()
    {
        vkDestroyBuffer(device, buffer, nullptr);

        if (bufferMemory != nullptr)
        {
            vkFreeMemory(device, bufferMemory, nullptr);
        }
    }

    VkMemoryRequirements getMemoryRequirements() const
    {
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        return requirements;
    }

    void allocateMemory(const VkMemoryPropertyFlags properties)
    {
        VkMemoryRequirements memoryRequirements = getMemoryRequirements();
        uint32_t memoryTypeIndex = physicalDevice->getCompatibleMemoryTypeIndex(memoryRequirements.memoryTypeBits, properties);

        const VkMemoryAllocateInfo memoryAllocateInfo =
        {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            memoryRequirements.size,
            memoryTypeIndex
        };

        checkVulkanError(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &bufferMemory), "vkAllocateMemory");
        checkVulkanError(vkBindBufferMemory(device, buffer, bufferMemory, 0), "vkBindBufferMemory");
    }

    void uploadData(const void* source, const VkDeviceSize dataSize)
    {
        void* data;
        checkVulkanError(vkMapMemory(device, bufferMemory, 0, dataSize, 0, &data), "vkMapMemory");
        std::memcpy(data, source, static_cast<size_t>(dataSize));
        vkUnmapMemory(device, bufferMemory);
    }

    void downloadData(void* target, const VkDeviceSize dataSize)
    {
        void* data;
        checkVulkanError(vkMapMemory(device, bufferMemory, 0, dataSize, 0, &data), "vkMapMemory");
        std::memcpy(target, data, static_cast<size_t>(dataSize));
        vkUnmapMemory(device, bufferMemory);
    }

    void recordCopyDataCommand(VkCommandBuffer commandBuffer, VkBuffer sourceBuffer, const VkDeviceSize dataSize)
    {
        const VkCommandBufferBeginInfo commandBufferBeginInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
        };

        const VkBufferCopy copyRegion =
        {
            0,
            0,
            dataSize
        };

        checkVulkanError(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo), "vkBeginCommandBuffer");
        vkCmdCopyBuffer(commandBuffer, sourceBuffer, buffer, 1, &copyRegion);
        checkVulkanError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    VkDevice getDevice() const
    {
        return device;
    }

    const VulkanPhysicalDevice& getPhysicalDevice() const
    {
        return *physicalDevice;
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    VkDeviceSize getBufferSize() const
    {
        return bufferSize;
    }

    VkBufferUsageFlags getUsageFlags() const
    {
        return usageFlags;
    }

    size_t getElementSize() const
    {
        return elementSize;
    }

    ArgumentId getKernelArgumentId() const
    {
        return kernelArgumentId;
    }

    ArgumentDataType getDataType() const
    {
        return dataType;
    }

    ArgumentMemoryLocation getMemoryLocation() const
    {
        return memoryLocation;
    }

    ArgumentAccessType getAccessType() const
    {
        return accessType;
    }

private:
    VkDevice device;
    const VulkanPhysicalDevice* physicalDevice;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    VkDeviceSize bufferSize;
    VkBufferUsageFlags usageFlags;
    size_t elementSize;
    ArgumentId kernelArgumentId;
    ArgumentDataType dataType;
    ArgumentMemoryLocation memoryLocation;
    ArgumentAccessType accessType;
};

} // namespace ktt
