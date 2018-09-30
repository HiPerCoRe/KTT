#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "vulkan/vulkan.h"
#include "vulkan_physical_device.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanDevice
{
public:
    VulkanDevice() :
        device(nullptr),
        queue(nullptr),
        queueType(VK_QUEUE_COMPUTE_BIT),
        queueFamilyIndex(0)
    {}

    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const VkQueueFlagBits queueType, const std::vector<const char*>& extensions) :
        VulkanDevice(physicalDevice, queueType, extensions, std::vector<const char*>{})
    {}

    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const VkQueueFlagBits queueType, const std::vector<const char*>& extensions,
        const std::vector<const char*>& validationLayers) :
        physicalDevice(physicalDevice),
        queueType(queueType)
    {
        if (!checkExtensionSupport(extensions))
        {
            throw std::runtime_error("One of the requested device extensions is not present");
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.getPhysicalDevice(), &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.getPhysicalDevice(), &queueFamilyCount, queueFamilies.data());

        bool queueFound = false;

        for (uint32_t i = 0; i < queueFamilies.size(); ++i)
        {
            if (queueFamilies.at(i).queueCount > 0 && queueFamilies.at(i).queueFlags & queueType)
            {
                queueFamilyIndex = i;
                queueFound = true;
                break;
            }
        }

        if (!queueFound)
        {
            throw std::runtime_error("Current device does not have any queues with specified queue type available");
        }

        const VkPhysicalDeviceFeatures deviceFeatures = {};
        const float queuePriority = 1.0f;
        const VkDeviceQueueCreateInfo deviceQueueCreateInfo =
        {
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            queueFamilyIndex,
            1,
            &queuePriority
        };

        const VkDeviceCreateInfo deviceCreateInfo =
        {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            1,
            &deviceQueueCreateInfo,
            static_cast<uint32_t>(validationLayers.size()),
            validationLayers.data(),
            static_cast<uint32_t>(extensions.size()),
            extensions.data(),
            &deviceFeatures
        };

        checkVulkanError(vkCreateDevice(physicalDevice.getPhysicalDevice(), &deviceCreateInfo, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    ~VulkanDevice()
    {
        waitIdle();
        vkDestroyDevice(device, nullptr);
    }

    const VulkanPhysicalDevice& getPhysicalDevice() const
    {
        return physicalDevice;
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkQueue getQueue() const
    {
        return queue;
    }

    VkQueueFlagBits getQueueType() const
    {
        return queueType;
    }

    uint32_t getQueueFamilyIndex() const
    {
        return queueFamilyIndex;
    }

    void waitIdle() const
    {
        checkVulkanError(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
    }

    uint32_t getSuitableMemoryTypeIndex(const uint32_t typeFilter, const VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            if (typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Current device does not have any suitable memory types available");
    }

    void queueSubmit(VkCommandBuffer commandBuffer)
    {
        const VkSubmitInfo submitInfo =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &commandBuffer,
            0,
            nullptr
        };

        checkVulkanError(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
        vkQueueWaitIdle(queue);
    }

private:
    VulkanPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkQueueFlagBits queueType;
    uint32_t queueFamilyIndex;

    bool checkExtensionSupport(const std::vector<const char*>& extensions)
    {
        std::vector<std::string> supportedExtensions = physicalDevice.getExtensions();

        for (const char* extension : extensions)
        {
            bool extensionsFound = false;
            for (const auto& comparedExtension : supportedExtensions)
            {
                if (std::string(extension) == comparedExtension)
                {
                    extensionsFound = true;
                    break;
                }
            }

            if (!extensionsFound)
            {
                return false;
            }
        }

        return true;
    }
};

} // namespace ktt
