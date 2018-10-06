#pragma once

#include <string>
#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanShaderModule
{
public:
    VulkanShaderModule() :
        device(nullptr),
        shaderModule(nullptr),
        source(nullptr)
    {}

    explicit VulkanShaderModule(VkDevice device, const std::string& source) :
        device(device),
        source(source)
    {
        const VkShaderModuleCreateInfo shaderModuleCreateInfo =
        {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            source.length(),
            reinterpret_cast<const uint32_t*>(source.data())
        };

        checkVulkanError(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule), "vkCreateShaderModule");
    }

    ~VulkanShaderModule()
    {
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkShaderModule getShaderModule() const
    {
        return shaderModule;
    }

    const std::string& getSource() const
    {
        return source;
    }

private:
    VkDevice device;
    VkShaderModule shaderModule;
    std::string source;
};

} // namespace ktt
