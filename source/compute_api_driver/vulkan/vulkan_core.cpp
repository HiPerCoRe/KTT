#include <stdexcept>

#include "vulkan_core.h"

namespace ktt
{

VulkanCore::VulkanCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{}

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

} // namespace ktt
