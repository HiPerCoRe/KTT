#pragma once

#ifdef KTT_API_VULKAN

#include <string>
#include <vulkan/vulkan.h>

namespace ktt
{

std::string GetEnumName(const VkResult value);
void CheckError(const VkResult value, const std::string& function, const std::string& info = "");

} // namespace ktt

#endif // KTT_API_VULKAN
