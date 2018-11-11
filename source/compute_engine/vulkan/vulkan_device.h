#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_physical_device.h>
#include <compute_engine/vulkan/vulkan_queue.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanDevice
{
public:
    VulkanDevice() :
        device(nullptr),
        queueCount(0),
        queueType(VK_QUEUE_COMPUTE_BIT),
        queueFamilyIndex(0)
    {}

    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType) :
        VulkanDevice(physicalDevice, queueCount, queueType, std::vector<const char*>{}, std::vector<const char*>{})
    {}

    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType,
        const std::vector<const char*>& extensions) :
        VulkanDevice(physicalDevice, queueCount, queueType, extensions, std::vector<const char*>{})
    {}

    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType,
        const std::vector<const char*>& extensions, const std::vector<const char*>& validationLayers) :
        physicalDevice(physicalDevice),
        queueCount(queueCount),
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
            queueCount,
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
        queues.resize(queueCount);

        for (uint32_t i = 0; i < queueCount; ++i)
        {
            vkGetDeviceQueue(device, queueFamilyIndex, i, &queues[i]);
        }
    }

    ~VulkanDevice()
    {
        if (device != nullptr)
        {
            waitIdle();
            vkDestroyDevice(device, nullptr);
        }
    }

    const VulkanPhysicalDevice& getPhysicalDevice() const
    {
        return physicalDevice;
    }

    VkDevice getDevice() const
    {
        return device;
    }

    uint32_t getQueueCount() const
    {
        return queueCount;
    }

    VkQueueFlagBits getQueueType() const
    {
        return queueType;
    }

    uint32_t getQueueFamilyIndex() const
    {
        return queueFamilyIndex;
    }

    std::vector<VulkanQueue> getQueues() const
    {
        std::vector<VulkanQueue> result;

        for (auto queue : queues)
        {
            result.emplace_back(queue, queueType);
        }

        return result;
    }

    void waitIdle() const
    {
        vkDeviceWaitIdle(device);
    }

private:
    VulkanPhysicalDevice physicalDevice;
    VkDevice device;
    std::vector<VkQueue> queues;
    uint32_t queueCount;
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
