#pragma once

#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VkMemAllocKtt.h>

namespace ktt
{

class VulkanDevice;
class VulkanInstance;

class VulkanMemoryAllocator
{
public:
    explicit VulkanMemoryAllocator(const VulkanInstance& instance, const VulkanDevice& device);
    ~VulkanMemoryAllocator();

    VmaAllocator GetAllocator() const;

private:
    VmaAllocator m_Allocator;
};

} // namespace ktt

#endif // KTT_API_VULKAN
