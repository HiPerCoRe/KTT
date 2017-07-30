#pragma once

#include <string>

#include "vulkan/vulkan.h"

namespace ktt
{

class VulkanPhysicalDevice
{
public:
    explicit VulkanPhysicalDevice(const VkPhysicalDevice physicalDevice, const std::string& name) :
        physicalDevice(physicalDevice),
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

private:
    VkPhysicalDevice physicalDevice;
    std::string name;
};

} // namespace ktt
