#ifdef KTT_API_VULKAN

#include <cstdio>

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanPhysicalDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

VulkanPhysicalDevice::VulkanPhysicalDevice(const DeviceIndex index, const VkPhysicalDevice device) :
    m_Index(index),
    m_Device(device)
{}

DeviceIndex VulkanPhysicalDevice::GetIndex() const
{
    return m_Index;
}

VkPhysicalDevice VulkanPhysicalDevice::GetPhysicalDevice() const
{
    return m_Device;
}

DeviceType VulkanPhysicalDevice::GetDeviceType() const
{
    const auto properties = GetProperties();

    switch (properties.deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        return DeviceType::Custom;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return DeviceType::GPU;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    default:
        KttError("Unhandled Vulkan physical device type value");
        return DeviceType::Custom;
    }
}

DeviceInfo VulkanPhysicalDevice::GetInfo() const
{
    const auto properties = GetProperties();
    const auto memoryProperties = GetMemoryProperties();
    
    DeviceInfo result(m_Index, properties.deviceName);
    result.SetVendor(std::to_string(properties.vendorID));

    // Persistent hardware identifier: the device UUID exposed through Vulkan 1.1 core (vkGetPhysicalDeviceProperties2).
    VkPhysicalDeviceIDProperties idProperties{};
    idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    // Driver version string (e.g. "550.54.14") exposed through Vulkan 1.2 core. Only chained when the device supports
    // Vulkan 1.2, otherwise the driver version stays empty.
    VkPhysicalDeviceDriverProperties driverProperties{};
    driverProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    const bool hasDriverProperties = properties.apiVersion >= VK_API_VERSION_1_2;

    if (hasDriverProperties)
    {
        idProperties.pNext = &driverProperties;
    }

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &idProperties;

    vkGetPhysicalDeviceProperties2(m_Device, &properties2);

    if (hasDriverProperties)
    {
        result.SetDriverVersion(driverProperties.driverInfo);
    }

    const auto* uuid = idProperties.deviceUUID;
    char identifier[37];
    std::snprintf(identifier, sizeof(identifier),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
        uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
    result.SetDeviceIdentifier(identifier);

    std::vector<std::string> extensions = GetExtensions();
    std::string mergedExtensions;

    for (size_t i = 0; i < extensions.size(); ++i)
    {
        mergedExtensions += extensions[i];

        if (i != extensions.size() - 1)
        {
            mergedExtensions += ", ";
        }
    }

    result.SetExtensions(mergedExtensions);
    result.SetDeviceType(GetDeviceType());

    bool memorySizeFound = false;

    for (const auto& heap : memoryProperties.memoryHeaps)
    {
        if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            result.SetGlobalMemorySize(heap.size);
            memorySizeFound = true;
            break;
        }
    }

    if (!memorySizeFound)
    {
        result.SetGlobalMemorySize(memoryProperties.memoryHeaps[0].size);
    }

    result.SetLocalMemorySize(static_cast<uint64_t>(properties.limits.maxComputeSharedMemorySize));
    result.SetMaxWorkGroupSize(static_cast<uint64_t>(properties.limits.maxComputeWorkGroupSize[0]));
    result.SetMaxConstantBufferSize(static_cast<uint64_t>(properties.limits.maxUniformBufferRange));

    // Todo: this info can be currently found only through specific HW vendor extensions
    result.SetMaxComputeUnits(0);

    result.SetCudaComputeCapabilityMajor(0);
    result.SetCudaComputeCapabilityMinor(0);

    return result;
}

VkPhysicalDeviceProperties VulkanPhysicalDevice::GetProperties() const
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(m_Device, &deviceProperties);
    return deviceProperties;
}

VkPhysicalDeviceMemoryProperties VulkanPhysicalDevice::GetMemoryProperties() const
{
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(m_Device, &deviceMemoryProperties);
    return deviceMemoryProperties;
}

std::vector<std::string> VulkanPhysicalDevice::GetExtensions() const
{
    uint32_t extensionCount;
    CheckError(vkEnumerateDeviceExtensionProperties(m_Device, nullptr, &extensionCount, nullptr),
        "vkEnumerateDeviceExtensionProperties");

    std::vector<VkExtensionProperties> extensions(static_cast<size_t>(extensionCount));
    CheckError(vkEnumerateDeviceExtensionProperties(m_Device, nullptr, &extensionCount, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");

    std::vector<std::string> result;

    for (const auto& extension : extensions)
    {
        result.emplace_back(extension.extensionName);
    }

    return result;
}

uint32_t VulkanPhysicalDevice::GetCompatibleMemoryTypeIndex(const uint32_t memoryTypeBits,
    const VkMemoryPropertyFlags properties) const
{
    const VkPhysicalDeviceMemoryProperties memoryProperties = GetMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if (memoryTypeBits & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw KttException("Physical device does not have any suitable memory types available");
}

} // namespace ktt

#endif // KTT_API_VULKAN
