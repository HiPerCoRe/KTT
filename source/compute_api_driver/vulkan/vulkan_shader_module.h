#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanShaderModule
{
public:
    explicit VulkanShaderModule(const VkDevice device, const std::string& source, const std::string& functionName) :
        device(device),
        functionName(functionName),
        argumentsCount(0)
    {
        /*VkShaderModuleCreateInfo shaderModuleCreateInfo =
        {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            sizeof(shader),
            shader
        };

        checkVulkanError(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule), "vkCreateShaderModule");*/
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

    std::string getFunctionName() const
    {
        return functionName;
    }

    size_t getArgumentsCount() const
    {
        return argumentsCount;
    }

private:
    VkDevice device;
    VkShaderModule shaderModule;
    std::string functionName;
    size_t argumentsCount;
};

} // namespace ktt
