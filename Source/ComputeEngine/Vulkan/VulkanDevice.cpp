#ifdef KTT_API_VULKAN

#include <string>

#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

VulkanDevice::VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType) :
    VulkanDevice(physicalDevice, queueCount, queueType, std::vector<const char*>{}, std::vector<const char*>{})
{}

VulkanDevice::VulkanDevice(const VulkanPhysicalDevice& physicalDevice, const uint32_t queueCount, const VkQueueFlagBits queueType,
    const std::vector<const char*>& extensions, const std::vector<const char*>& validationLayers) :
    m_PhysicalDevice(physicalDevice),
    m_QueueCount(queueCount),
    m_QueueType(queueType)
{
    Logger::LogDebug("Initializing Vulkan device");
    KttAssert(CheckExtensions(extensions), "Some of the requested device extensions are not present");

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.GetPhysicalDevice(), &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(static_cast<size_t>(queueFamilyCount));
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.GetPhysicalDevice(), &queueFamilyCount, queueFamilies.data());

    bool queueFound = false;

    for (uint32_t i = 0; i < queueFamilies.size(); ++i)
    {
        if (queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags & queueType)
        {
            m_QueueFamilyIndex = i;
            queueFound = true;
            break;
        }
    }

    if (!queueFound)
    {
        throw KttException("Current device does not have any queues with the specified queue type available");
    }

    const VkPhysicalDeviceFeatures deviceFeatures = {};
    const std::vector<float> queuePriorities(static_cast<size_t>(queueCount), 1.0f);
    const VkDeviceQueueCreateInfo queueCreateInfo =
    {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        m_QueueFamilyIndex,
        queueCount,
        queuePriorities.data()
    };

    const VkDeviceCreateInfo deviceCreateInfo =
    {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
        0,
        1,
        &queueCreateInfo,
        static_cast<uint32_t>(validationLayers.size()),
        validationLayers.data(),
        static_cast<uint32_t>(extensions.size()),
        extensions.data(),
        &deviceFeatures
    };

    CheckError(vkCreateDevice(physicalDevice.GetPhysicalDevice(), &deviceCreateInfo, nullptr, &m_Device), "vkCreateDevice");
}

VulkanDevice::~VulkanDevice()
{
    Logger::LogDebug("Releasing Vulkan device");
    WaitIdle();
    vkDestroyDevice(m_Device, nullptr);
}

const VulkanPhysicalDevice& VulkanDevice::GetPhysicalDevice() const
{
    return m_PhysicalDevice;
}

VkDevice VulkanDevice::GetDevice() const
{
    return m_Device;
}

std::vector<VkQueue> VulkanDevice::GetQueues() const
{
    std::vector<VkQueue> result(static_cast<size_t>(m_QueueCount));

    for (uint32_t i = 0; i < m_QueueCount; ++i)
    {
        vkGetDeviceQueue(m_Device, m_QueueFamilyIndex, i, &result[i]);
    }

    return result;
}

VkQueueFlagBits VulkanDevice::GetQueueType() const
{
    return m_QueueType;
}

uint32_t VulkanDevice::GetQueueFamilyIndex() const
{
    return m_QueueFamilyIndex;
}

void VulkanDevice::WaitIdle() const
{
    Logger::LogDebug("Waiting for Vulkan device to be idle");
    vkDeviceWaitIdle(m_Device);
}

bool VulkanDevice::CheckExtensions(const std::vector<const char*>& extensions)
{
    std::vector<std::string> supportedExtensions = m_PhysicalDevice.GetExtensions();

    for (const char* extension : extensions)
    {
        const bool extensionFound = ContainsElementIf(supportedExtensions, [&extension](const auto& currentExtension)
        {
            return std::string(extension) == currentExtension;
        });

        if (!extensionFound)
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt

#endif // KTT_API_VULKAN
