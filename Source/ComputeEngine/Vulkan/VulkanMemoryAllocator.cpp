#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanInstance.h>
#include <ComputeEngine/Vulkan/VulkanMemoryAllocator.h>
#include <ComputeEngine/Vulkan/VulkanPhysicalDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanMemoryAllocator::VulkanMemoryAllocator(const VulkanInstance& instance, const VulkanDevice& device)
{
    Logger::LogDebug("Initializing Vulkan memory allocator");

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = device.GetPhysicalDevice().GetPhysicalDevice();
    allocatorInfo.device = device.GetDevice();
    allocatorInfo.instance = instance.GetInstance();
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;

    CheckError(vmaCreateAllocator(&allocatorInfo, &m_Allocator), "vmaCreateAllocator");
}

VulkanMemoryAllocator::~VulkanMemoryAllocator()
{
    Logger::LogDebug("Releasing Vulkan memory allocator");
    vmaDestroyAllocator(m_Allocator);
}

VmaAllocator VulkanMemoryAllocator::GetAllocator() const
{
    return m_Allocator;
}

} // namespace ktt

#endif // KTT_API_VULKAN
