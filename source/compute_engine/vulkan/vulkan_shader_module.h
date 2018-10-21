#pragma once

#include <string>
#include "vulkan/vulkan.h"
#include "glslang_compiler.h"
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

    static std::string getTestSource()
    {
        const std::string result(std::string("")
            + "#version 450\n"
            + "#extension GL_ARB_separate_shader_objects : enable\n"
            + "layout(local_size_x_id = 0, local_size_y_id = 1) in; // Workgroup size defined with specialization constants, on cpp side there is associated SpecializationInfo entry in PipelineShaderStageCreateInfo\n"
            + "layout(push_constant) uniform Parameters {           // Specify push constants, on cpp side its layout is fixed at PipelineLayout, and values are provided via vk::CommandBuffer::pushConstants()\n"
            + "uint Width;\n"
            + "uint Height;\n"
            + "float a;\n"
            + "} params;\n"
            + "\n"
            + "layout(std430, binding = 0) buffer lay0 { float arr_y[]; };\n"
            + "layout(std430, binding = 1) buffer lay1 { float arr_x[]; };\n"
            + "\n"
            + "void main() {\n"
            + "    // Drop threads outside the buffer dimensions\n"
            + "    if (params.Width <= gl_GlobalInvocationID.x || params.Height <= gl_GlobalInvocationID.y) {\n"
            + "        return;"
            + "    }\n"
            + "    const uint id = params.Width*gl_GlobalInvocationID.y + gl_GlobalInvocationID.x; // current offset\n"
            + "    \n"
            + "    arr_y[id] += params.a*arr_x[id]; // saxpy\n"
            + "}\n");

        return result;
    }

private:
    VkDevice device;
    VkShaderModule shaderModule;
    std::string source;
    std::vector<uint32_t> spirvSource;
};

} // namespace ktt
