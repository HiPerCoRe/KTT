#ifdef KTT_API_VULKAN

#include <cstdint>
#include <iostream>

#include <ComputeEngine/Vulkan/VulkanInstance.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

VulkanInstance::VulkanInstance(const std::string& applicationName) :
    VulkanInstance(applicationName, std::vector<const char*>{}, std::vector<const char*>{})
{}

VulkanInstance::VulkanInstance(const std::string& applicationName, const std::vector<const char*>& extensions,
    const std::vector<const char*>& validationLayers) :
    m_DebugCallback(nullptr)
{
    KttAssert(CheckValidationLayers(validationLayers), "Some of the requested validation layers are not present");

    const VkApplicationInfo applicationInfo =
    {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        applicationName.c_str(),
        VK_MAKE_VERSION(1, 0, 0),
        "KTT",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_0
    };

    const VkInstanceCreateInfo instanceInfo =
    {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &applicationInfo,
        static_cast<uint32_t>(validationLayers.size()),
        validationLayers.data(),
        static_cast<uint32_t>(extensions.size()),
        extensions.data()
    };

    CheckError(vkCreateInstance(&instanceInfo, nullptr, &m_Instance), "vkCreateInstance");

    if (!validationLayers.empty())
    {
        InitializeDebugCallback();
    }
}

VulkanInstance::~VulkanInstance()
{
    if (m_DebugCallback != nullptr)
    {
        auto destroyDebugReportCallbackEXT = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(m_Instance,
            "vkDestroyDebugReportCallbackEXT"));
        destroyDebugReportCallbackEXT(m_Instance, m_DebugCallback, nullptr);
    }

    vkDestroyInstance(m_Instance, nullptr);
}

VkInstance VulkanInstance::GetInstance() const
{
    return m_Instance;
}

std::vector<VulkanPhysicalDevice> VulkanInstance::GetPhysicalDevices() const
{
    uint32_t deviceCount;
    CheckError(vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");

    std::vector<VkPhysicalDevice> devices(static_cast<size_t>(deviceCount));
    CheckError(vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

    std::vector<VulkanPhysicalDevice> result;

    for (uint32_t i = 0; i < deviceCount; ++i)
    {
        result.emplace_back(static_cast<DeviceIndex>(i), devices[i]);
    }

    return result;
}

std::string VulkanInstance::GetApiVersion() const
{
    uint32_t encodedVersion;
    CheckError(vkEnumerateInstanceVersion(&encodedVersion), "vkEnumerateInstanceVersion");

    std::string result;
    result += std::to_string(VK_VERSION_MAJOR(encodedVersion)) + "." + std::to_string(VK_VERSION_MINOR(encodedVersion)) + "."
        + std::to_string(VK_VERSION_PATCH(encodedVersion));
    return result;
}

std::vector<std::string> VulkanInstance::GetExtensions() const
{
    uint32_t extensionCount;
    CheckError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "vkEnumerateInstanceExtensionProperties");

    std::vector<VkExtensionProperties> extensions(static_cast<size_t>(extensionCount));
    CheckError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()),
        "vkEnumerateInstanceExtensionProperties");

    std::vector<std::string> result;

    for (const auto& extension : extensions)
    {
        result.emplace_back(extension.extensionName);
    }

    return result;
}

void VulkanInstance::InitializeDebugCallback()
{
    const VkDebugReportCallbackCreateInfoEXT createInfo =
    {
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
        nullptr,
        VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
        DebugCallback,
        nullptr
    };

    auto createDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(m_Instance,
        "vkCreateDebugReportCallbackEXT"));
    createDebugReportCallbackEXT(m_Instance, &createInfo, nullptr, &m_DebugCallback);
}

VkBool32 VulkanInstance::DebugCallback([[maybe_unused]] VkDebugReportFlagsEXT flags,
    [[maybe_unused]] VkDebugReportObjectTypeEXT objectType, [[maybe_unused]] uint64_t object, [[maybe_unused]] size_t location,
    [[maybe_unused]] int32_t code, [[maybe_unused]] const char* layerPrefix, const char* message, [[maybe_unused]] void* userData)
{
    Logger::LogError(std::string("Vulkan validation layer error: ") + message);
    return VK_FALSE;
}

bool VulkanInstance::CheckValidationLayers(const std::vector<const char*>& validationLayers)
{
    uint32_t layerCount;
    CheckError(vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "vkEnumerateInstanceLayerProperties");

    std::vector<VkLayerProperties> availableLayers(static_cast<size_t>(layerCount));
    CheckError(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()), "vkEnumerateInstanceLayerProperties");

    for (const char* requiredLayer : validationLayers)
    {
        const bool layerFound = ContainsElementIf(availableLayers, [&requiredLayer](const auto& layer)
        {
            return std::string(requiredLayer) == std::string(layer.layerName);
        });

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt

#endif // KTT_API_VULKAN
