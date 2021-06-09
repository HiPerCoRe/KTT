#pragma once

#ifdef KTT_API_VULKAN

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <Api/Info/DeviceInfo.h>
#include <KttTypes.h>

namespace ktt
{

class VulkanPhysicalDevice
{
public:
    explicit VulkanPhysicalDevice(const DeviceIndex index, const VkPhysicalDevice device);

    DeviceIndex GetIndex() const;
    VkPhysicalDevice GetPhysicalDevice() const;
    DeviceType GetDeviceType() const;
    DeviceInfo GetInfo() const;
    VkPhysicalDeviceProperties GetProperties() const;
    VkPhysicalDeviceMemoryProperties GetMemoryProperties() const;
    std::vector<std::string> GetExtensions() const;
    uint32_t GetCompatibleMemoryTypeIndex(const uint32_t memoryTypeBits, const VkMemoryPropertyFlags properties) const;

private:
    DeviceIndex m_Index;
    VkPhysicalDevice m_Device;
};

} // namespace ktt

#endif // KTT_API_VULKAN
