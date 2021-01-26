#pragma once

#ifdef KTT_API_VULKAN

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/VulkanPhysicalDevice.h>

namespace ktt
{

class VulkanInstance
{
public:
    explicit VulkanInstance(const std::string& applicationName);
    explicit VulkanInstance(const std::string& applicationName, const std::vector<const char*>& extensions,
        const std::vector<const char*>& validationLayers);
    ~VulkanInstance();

    VkInstance GetInstance() const;
    std::vector<VulkanPhysicalDevice> GetPhysicalDevices() const;
    std::string GetApiVersion() const;
    std::vector<std::string> GetExtensions() const;

private:
    VkInstance m_Instance;
    VkDebugReportCallbackEXT m_DebugCallback;

    void InitializeDebugCallback();

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
        uint64_t object, size_t location, int32_t code, const char* layerPrefix, const char* message, void* userData);
    static bool CheckValidationLayers(const std::vector<const char*>& validationLayers);
};

} // namespace ktt

#endif // KTT_API_VULKAN
