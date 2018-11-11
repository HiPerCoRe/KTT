#pragma once

#ifdef KTT_PLATFORM_VULKAN

#include <string>
#include <vulkan/vulkan.h>

namespace ktt
{

std::string getVulkanEnumName(const VkResult value);
void checkVulkanError(const VkResult value);
void checkVulkanError(const VkResult value, const std::string& message);

} // namespace ktt

#endif // KTT_PLATFORM_VULKAN
