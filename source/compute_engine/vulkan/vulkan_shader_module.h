#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <shaderc/shaderc.hpp>
#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanShaderModule
{
public:
    explicit VulkanShaderModule(const VkDevice device, const std::string& shaderName, const std::string& shaderSource) :
        device(device),
        shaderName(shaderName),
        shaderSource(shaderSource)
    {
        std::string preprocessedSource = preprocessShader(shaderName, shaderc_compute_shader, shaderSource);
        spirvSource = compileSource(shaderName, shaderc_compute_shader, preprocessedSource, true);

        VkShaderModuleCreateInfo shaderModuleCreateInfo =
        {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            spirvSource.size() * 4, // in bytes
            spirvSource.data()
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
    
    std::string getShaderName() const
    {
        return shaderName;
    }

    std::string getShaderSource() const
    {
        return shaderSource;
    }

    VkShaderModule getShaderModule() const
    {
        return shaderModule;
    }

    std::vector<uint32_t> getSpirvSource() const
    {
        return spirvSource;
    }

private:
    VkDevice device;
    std::string shaderName;
    std::string shaderSource;
    VkShaderModule shaderModule;
    std::vector<uint32_t> spirvSource;

    std::string preprocessShader(const std::string& shaderName, const shaderc_shader_kind kind, const std::string& source)
    {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        shaderc::PreprocessedSourceCompilationResult result = compiler.PreprocessGlsl(source, kind, shaderName.c_str(), options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Internal Vulkan error: ") + result.GetErrorMessage());
        }

        return {result.cbegin(), result.cend()};
    }

    std::vector<uint32_t> compileSource(const std::string& shaderName, const shaderc_shader_kind kind, const std::string& source,
        const bool optimize)
    {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        if (optimize)
        {
            options.SetOptimizationLevel(shaderc_optimization_level_size);
        }

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, kind, shaderName.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Internal Vulkan error: ") + module.GetErrorMessage());
        }

        return {module.cbegin(), module.cend()};
    }
};

} // namespace ktt
