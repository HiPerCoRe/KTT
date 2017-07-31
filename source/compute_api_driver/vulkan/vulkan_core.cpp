#include <cstdint>
#include <stdexcept>

#include "vulkan_core.h"

namespace ktt
{

#ifdef PLATFORM_VULKAN

VulkanCore::VulkanCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{
    auto devices = getVulkanDevices();
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    device = std::make_unique<VulkanDevice>(devices.at(deviceIndex).getPhysicalDevice());
    queue = std::make_unique<VulkanQueue>(device->getDevice(), device->getComputeQueueIndices().at(0));
    commandPool = std::make_unique<VulkanCommandPool>(device->getDevice(), queue->getQueueIndex());
    commandBuffer = std::make_unique<VulkanCommandBuffer>(device->getDevice(), commandPool->getCommandPool());
}

void VulkanCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    outputTarget << "Platform 0: " << "Vulkan" << std::endl;
    auto devices = getVulkanDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> VulkanCore::getPlatformInfo() const
{
    PlatformInfo vulkan(0, "Vulkan");
    vulkan.setVendor("");
    vulkan.setVersion("1.0.51");
    vulkan.setExtensions("");
    return std::vector<PlatformInfo>{ vulkan };
}

std::vector<DeviceInfo> VulkanCore::getDeviceInfo(const size_t platformIndex) const
{
    std::vector<DeviceInfo> result;
    auto devices = getVulkanDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getVulkanDeviceInfo(i));
    }

    return result;
}

DeviceInfo VulkanCore::getCurrentDeviceInfo() const
{
    return getVulkanDeviceInfo(deviceIndex);
}

void VulkanCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void VulkanCore::uploadArgument(const KernelArgument& kernelArgument)
{
    throw std::runtime_error("uploadArgument() method is not supported for Vulkan yet");
}

void VulkanCore::updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes)
{
    throw std::runtime_error("updateArgument() method is not supported for Vulkan yet");
}

KernelArgument VulkanCore::downloadArgument(const size_t argumentId) const
{
    throw std::runtime_error("downloadArgument() method is not supported for Vulkan yet");
}

void VulkanCore::clearBuffer(const size_t argumentId)
{
    throw std::runtime_error("clearBuffer() method is not supported for Vulkan yet");
}

void VulkanCore::clearBuffers()
{
    throw std::runtime_error("clearBuffers() method is not supported for Vulkan yet");
}

void VulkanCore::clearBuffers(const ArgumentMemoryType& argumentMemoryType)
{
    throw std::runtime_error("clearBuffers() method is not supported for Vulkan yet");
}

KernelRunResult VulkanCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers)
{
    throw std::runtime_error("runKernel() method is not supported for Vulkan yet");
}

DeviceInfo VulkanCore::getVulkanDeviceInfo(const size_t deviceIndex) const
{
    auto devices = getVulkanDevices();
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(devices.at(deviceIndex).getPhysicalDevice(), &deviceProperties);

    result.setExtensions("");
    result.setVendor(std::to_string(deviceProperties.vendorID));
    result.setDeviceType(getDeviceType(deviceProperties.deviceType));

    result.setGlobalMemorySize(0);
    result.setLocalMemorySize(deviceProperties.limits.maxComputeSharedMemorySize);
    result.setMaxWorkGroupSize(0);
    result.setMaxConstantBufferSize(0);
    result.setMaxComputeUnits(0);

    return result;
}

std::vector<VulkanPhysicalDevice> VulkanCore::getVulkanDevices() const
{
    uint32_t deviceCount;
    checkVulkanError(vkEnumeratePhysicalDevices(vulkanInstance.getInstance(), &deviceCount, nullptr), "vkEnumeratePhysicalDevices");

    std::vector<VkPhysicalDevice> vulkanDevices(deviceCount);
    checkVulkanError(vkEnumeratePhysicalDevices(vulkanInstance.getInstance(), &deviceCount, vulkanDevices.data()), "vkEnumeratePhysicalDevices");

    std::vector<VulkanPhysicalDevice> devices;
    for (const auto vulkanDevice : vulkanDevices)
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(vulkanDevice, &deviceProperties);
        devices.push_back(VulkanPhysicalDevice(vulkanDevice, deviceProperties.deviceName));
    }

    return devices;
}

DeviceType VulkanCore::getDeviceType(const VkPhysicalDeviceType deviceType)
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

#else

VulkanCore::VulkanCore(const size_t)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::printComputeApiInfo(std::ostream&) const
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

std::vector<PlatformInfo> VulkanCore::getPlatformInfo() const
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

std::vector<DeviceInfo> VulkanCore::getDeviceInfo(const size_t) const
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

DeviceInfo VulkanCore::getCurrentDeviceInfo() const
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::setCompilerOptions(const std::string&)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::uploadArgument(const KernelArgument&)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::updateArgument(const size_t, const void*, const size_t)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

KernelArgument VulkanCore::downloadArgument(const size_t) const
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::clearBuffer(const size_t)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::clearBuffers()
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

void VulkanCore::clearBuffers(const ArgumentMemoryType&)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

KernelRunResult VulkanCore::runKernel(const std::string&, const std::string&, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<const KernelArgument*>&)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT library");
}

#endif // PLATFORM_VULKAN

} // namespace ktt
