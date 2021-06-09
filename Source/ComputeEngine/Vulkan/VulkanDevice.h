#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/VulkanPhysicalDevice.h>

namespace ktt
{

class VulkanDevice
{
public:
    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType);
    explicit VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType,
        const std::vector<const char*>& extensions, const std::vector<const char*>& validationLayers);
    ~VulkanDevice();

    const VulkanPhysicalDevice& GetPhysicalDevice() const;
    VkDevice GetDevice() const;
    std::vector<VkQueue> GetQueues() const;
    VkQueueFlagBits GetQueueType() const;
    uint32_t GetQueueFamilyIndex() const;

    void WaitIdle() const;

private:
    VulkanPhysicalDevice m_PhysicalDevice;
    VkDevice m_Device;
    uint32_t m_QueueCount;
    VkQueueFlagBits m_QueueType;
    uint32_t m_QueueFamilyIndex;

    bool CheckExtensions(const std::vector<const char*>& extensions);
};

} // namespace ktt

#endif // KTT_API_VULKAN
