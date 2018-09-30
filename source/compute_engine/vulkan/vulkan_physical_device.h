#pragma once

#include <string>
#include "vulkan/vulkan.h"
#include "ktt_types.h"
#include "api/device_info.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanPhysicalDevice
{
public:
    VulkanPhysicalDevice() :
        VulkanPhysicalDevice(nullptr, 0, "")
    {}

    explicit VulkanPhysicalDevice(const VkPhysicalDevice physicalDevice, const DeviceIndex index, const std::string& name) :
        physicalDevice(physicalDevice),
        index(index),
        name(name)
    {}

    VkPhysicalDevice getPhysicalDevice() const
    {
        return physicalDevice;
    }

    std::string getName() const
    {
        return name;
    }

    DeviceInfo getDeviceInfo() const
    {
        DeviceInfo result(index, name);

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

        std::vector<std::string> extensions = getExtensions();
        std::string mergedExtensions("");
        for (size_t i = 0; i < extensions.size(); ++i)
        {
            mergedExtensions += extensions[i];
            if (i != extensions.size() - 1)
            {
                mergedExtensions += ", ";
            }
        }
        result.setExtensions(mergedExtensions);
        result.setVendor(std::to_string(deviceProperties.vendorID));
        result.setDeviceType(getDeviceType(deviceProperties.deviceType));

        bool memorySizeFound = false;
        for (const auto& heap : deviceMemoryProperties.memoryHeaps)
        {
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                result.setGlobalMemorySize(heap.size);
                memorySizeFound = true;
                break;
            }
        }
        if (!memorySizeFound)
        {
            result.setGlobalMemorySize(deviceMemoryProperties.memoryHeaps[0].size);
        }

        result.setLocalMemorySize(deviceProperties.limits.maxComputeSharedMemorySize);
        result.setMaxWorkGroupSize(deviceProperties.limits.maxComputeWorkGroupSize[0]);
        result.setMaxConstantBufferSize(deviceProperties.limits.maxUniformBufferRange);
        result.setMaxComputeUnits(0); // to do: find this information for Vulkan API

        return result;
    }

    VkPhysicalDeviceMemoryProperties getMemoryProperties() const
    {
        VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
        return deviceMemoryProperties;
    }

    std::vector<std::string> getExtensions() const
    {
        uint32_t extensionCount;
        checkVulkanError(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr),
            "vkEnumerateDeviceExtensionProperties");

        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

        std::vector<std::string> result;
        for (const auto& extension : extensions)
        {
            result.emplace_back(extension.extensionName);
        }

        return result;
    }

    static DeviceType getDeviceType(const VkPhysicalDeviceType deviceType)
    {
        switch (deviceType)
        {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            return DeviceType::Custom;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return DeviceType::GPU;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return DeviceType::GPU;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return DeviceType::GPU;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return DeviceType::CPU;
        default:
            return DeviceType::Custom;
        }
    }

private:
    VkPhysicalDevice physicalDevice;
    DeviceIndex index;
    std::string name;
};

} // namespace ktt
