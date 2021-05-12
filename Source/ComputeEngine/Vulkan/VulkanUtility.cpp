#ifdef KTT_API_VULKAN

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

std::string GetEnumName(const VkResult value)
{
    switch (value)
    {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_NOT_READY:
        return "VK_NOT_READY";
    case VK_TIMEOUT:
        return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
    default:
        return std::to_string(static_cast<int>(value));
    }
}

void CheckError(const VkResult value, const std::string& function, const std::string& info)
{
    if (value == VK_SUCCESS)
    {
        return;
    }

    std::string message = "Vulkan engine encountered error " + GetEnumName(value) + " in function " + function;

    if (!info.empty())
    {
        message += ", additional info: " + info;
    }

    throw KttException(message);
}

} // namespace ktt

#endif // KTT_PLATFORM_VULKAN
