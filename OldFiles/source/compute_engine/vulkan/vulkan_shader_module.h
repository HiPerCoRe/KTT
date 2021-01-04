#pragma once

#include <string>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/shaderc_compiler.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanShaderModule
{
public:
    VulkanShaderModule() :
        device(nullptr),
        shaderModule(nullptr),
        name(""),
        source("")
    {}

    explicit VulkanShaderModule(VkDevice device, const std::string& name, const std::string& source, const std::vector<size_t>& localSize,
        const std::vector<ParameterPair>& parameterPairs) :
        device(device),
        name(name),
        source(source)
    {
        spirvSource = ShadercCompiler::getCompiler().compile(name, source, shaderc_compute_shader, localSize, parameterPairs);

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

    const std::string& getName() const
    {
        return name;
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
    std::string name;
    std::string source;
    std::vector<uint32_t> spirvSource;
};

} // namespace ktt
