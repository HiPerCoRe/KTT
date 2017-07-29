#include <stdexcept>

#include "vulkan_core.h"

namespace ktt
{

#ifdef PLATFORM_VULKAN

VulkanCore::VulkanCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{
    vulkanInstance.initialize();
}

void VulkanCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    throw std::runtime_error("printComputeApiInfo() method is not supported for Vulkan yet");
}

std::vector<PlatformInfo> VulkanCore::getPlatformInfo() const
{
    throw std::runtime_error("getPlatformInfo() method is not supported for Vulkan yet");
}

std::vector<DeviceInfo> VulkanCore::getDeviceInfo(const size_t platformIndex) const
{
    throw std::runtime_error("getDeviceInfo() method is not supported for Vulkan yet");
}

DeviceInfo VulkanCore::getCurrentDeviceInfo() const
{
    throw std::runtime_error("getCurrentDeviceInfo() method is not supported for Vulkan yet");
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
