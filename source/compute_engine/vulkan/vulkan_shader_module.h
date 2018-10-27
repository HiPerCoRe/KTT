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

    static const std::string& getTestSource()
    {
        static const std::string result(std::string("")
            + "#version 450\n"
            + "#extension GL_ARB_separate_shader_objects : enable\n"
            + "#define WIDTH 3200\n"
            + "#define HEIGHT 2400\n"
            + "#define WORKGROUP_SIZE 32\n"
            + "layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;\n"
            + "layout(std140, binding = 0) buffer buf\n"
            + "{\n"
            + "    vec4 imageData[];\n"
            + "};\n"
            + "void main()\n"
            + "{\n"
            + "    if (gl_GlobalInvocationID.x >= WIDTH || gl_GlobalInvocationID.y >= HEIGHT)\n"
            + "        return;\n"
            + "    float x = float(gl_GlobalInvocationID.x) / float(WIDTH);\n"
            + "    float y = float(gl_GlobalInvocationID.y) / float(HEIGHT);\n"
            + "    vec2 uv = vec2(x, y);\n"
            + "    float n = 0.0f;\n"
            + "    vec2 c = vec2(-0.445f, 0.0f) + (uv - 0.5f) * (2.0f + 1.7f * 0.2f);\n"
            + "    vec2 z = vec2(0.0f);\n"
            + "    const int M = 128;\n"
            + "    for (int i = 0; i < M; ++i)\n"
            + "    {\n"
            + "        z = vec2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;\n"
            + "        if (dot(z, z) > 2.0f) break;\n"
            + "        n++;\n"
            + "    }\n"
            + "    float t = float(n) / float(M);\n"
            + "    vec3 d = vec3(0.3f, 0.3f, 0.5f);\n"
            + "    vec3 e = vec3(-0.2f, -0.3f, -0.5f);\n"
            + "    vec3 f = vec3(2.1f, 2.0f, 3.0f);\n"
            + "    vec3 g = vec3(0.0f, 0.1f, 0.0f);\n"
            + "    vec4 color = vec4(d + e * cos(6.28318f * (f * t + g)), 1.0f);\n"
            + "    imageData[WIDTH * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x] = color;\n"
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
