#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_physical_device.h>
#include <compute_engine/vulkan/vulkan_utility.h>
#include <utility/logger.h>

namespace ktt
{

class VulkanInstance
{
public:
    VulkanInstance() :
        instance(nullptr),
        callback(nullptr),
        debugCallbackLoaded(false)
    {}

    explicit VulkanInstance(const std::string& applicationName) :
        VulkanInstance(applicationName, std::vector<const char*>{})
    {}

    explicit VulkanInstance(const std::string& applicationName, const std::vector<const char*>& extensions) :
        VulkanInstance(applicationName, extensions, std::vector<const char*>{})
    {}

    explicit VulkanInstance(const std::string& applicationName, const std::vector<const char*>& extensions,
        const std::vector<const char*>& validationLayers) :
        debugCallbackLoaded(false)
    {
        if (!checkValidationLayerSupport(validationLayers))
        {
            throw std::runtime_error("One of the requested validation layers is not present");
        }

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

        const VkInstanceCreateInfo instanceCreateInfo =
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

        checkVulkanError(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "vkCreateInstance");

        if (validationLayers.size() != 0)
        {
            setupDebugCallback();
        }
    }

    ~VulkanInstance()
    {
        if (debugCallbackLoaded)
        {
            auto destroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance,
                "vkDestroyDebugReportCallbackEXT");
            destroyDebugReportCallbackEXT(instance, callback, nullptr);
        }

        if (instance != nullptr)
        {
            vkDestroyInstance(instance, nullptr);
        }
    }

    VkInstance getInstance() const
    {
        return instance;
    }

    std::vector<VulkanPhysicalDevice> getPhysicalDevices() const
    {
        uint32_t deviceCount;
        checkVulkanError(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        checkVulkanError(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

        std::vector<VulkanPhysicalDevice> result;
        for (uint32_t i = 0; i < deviceCount; ++i)
        {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(devices[i], &properties);
            result.emplace_back(devices[i], i, properties.deviceName);
        }

        return result;
    }

    std::string getAPIVersion() const
    {
        uint32_t encodedVersion;
        vkEnumerateInstanceVersion(&encodedVersion);

        std::string result("");
        result += std::to_string(VK_VERSION_MAJOR(encodedVersion)) + "." + std::to_string(VK_VERSION_MINOR(encodedVersion)) + "."
            + std::to_string(VK_VERSION_PATCH(encodedVersion));
        return result;
    }

    std::vector<std::string> getExtensions() const
    {
        uint32_t extensionCount;
        checkVulkanError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "vkEnumerateInstanceExtensionProperties");

        std::vector<VkExtensionProperties> extensions(extensionCount);
        checkVulkanError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()),
            "vkEnumerateInstanceExtensionProperties");

        std::vector<std::string> result;
        for (const auto& extension : extensions)
        {
            result.emplace_back(extension.extensionName);
        }

        return result;
    }

private:
    VkInstance instance;
    VkDebugReportCallbackEXT callback;
    bool debugCallbackLoaded;

    void setupDebugCallback()
    {
        const VkDebugReportCallbackCreateInfoEXT createInfo =
        {
            VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            nullptr,
            VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
            debugCallback,
            nullptr
        };

        auto createDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance,
            "vkCreateDebugReportCallbackEXT");
        createDebugReportCallbackEXT(instance, &createInfo, nullptr, &callback);
        debugCallbackLoaded = true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT /* flags */, VkDebugReportObjectTypeEXT /* objectType */,
        uint64_t /* object */, size_t /* location */, int32_t /* code */, const char* /* layerPrefix */, const char* message, void* /* userData */)
    {
        Logger::logError(std::string("Validation layer report: ") + message);
        return VK_FALSE;
    }

    static bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
    {
        uint32_t layerCount;
        checkVulkanError(vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "vkEnumerateInstanceLayerProperties");

        std::vector<VkLayerProperties> availableLayers(layerCount);
        checkVulkanError(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()), "vkEnumerateInstanceLayerProperties");

        for (const char* layer : validationLayers)
        {
            bool layerFound = false;
            for (const auto& comparedLayer : availableLayers)
            {
                if (std::string(layer) == std::string(comparedLayer.layerName))
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }
};

} // namespace ktt
