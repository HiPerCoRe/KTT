#pragma once

#include <string>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/glslang_compiler.h>
#include <compute_engine/vulkan/vulkan_utility.h>

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
        spirvSource = GlslangCompiler::getCompiler().compile(source, EShLangCompute);

        const VkShaderModuleCreateInfo shaderModuleCreateInfo =
        {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            spirvSource.size() * sizeof(uint32_t),
            spirvSource.data()
        };

        checkVulkanError(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule), "vkCreateShaderModule");
    }

    ~VulkanShaderModule()
    {
        if (shaderModule != nullptr)
        {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
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

    const std::vector<uint32_t>& getSpirvSource() const
    {
        return spirvSource;
    }

private:
    VkDevice device;
    VkShaderModule shaderModule;
    std::string source;
    std::vector<uint32_t> spirvSource;
};

} // namespace ktt
